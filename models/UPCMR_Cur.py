from typing import List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models.utils as utils
import random
import copy
import matplotlib.pyplot as plt
import fastmri

class Conv2_1d(nn.Module):
    def __init__(self, in_chans, chans, kernel_size=(3, 2, 2), stride=(1, 2, 2),
                        padding=(0, 0, 0), dilation=(1, 1, 1), bias=True, tcp=True):
        super().__init__()
        self.tcp = tcp
        self.conv2d = nn.Conv2d(in_chans, chans, kernel_size[1:], stride[1:], padding[1:], dilation=dilation[1:], bias=bias)
        if self.tcp:
            self.conv1d = nn.Conv1d(chans, chans, kernel_size[0], stride[0], padding[0], dilation=dilation[0], bias=bias)
        else:
            self.conv1d = nn.Conv1d(chans, chans, kernel_size[0], stride[0], 1, dilation=dilation[0], bias=bias)

    def forward(self, x):
        #2D convolution
        b=1
        t, c, d1, d2 = x.shape
        x = F.relu(self.conv2d(x))
        
        #1D convolution
        c, dr1, dr2 = x.size(1), x.size(2), x.size(3)
        x = x.view(b, t, c, dr1, dr2)
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(b*dr1*dr2, c, t)
        if self.tcp:
            x = torch.cat((x[:,:,t-1:t], x, x[:,:,0:1]), dim=-1)
        x = self.conv1d(x)

        #Final output
        out_c, out_t = x.size(1), x.size(2)
        x = x.view(b, dr1, dr2, out_c, out_t)
        x = x.permute(0, 4, 3, 1, 2).contiguous().reshape(-1, out_c, dr1, dr2)
        return x

class ConvTranspose2_1d(nn.Module):
    def __init__(self, in_chans, chans, kernel_size=(3, 2, 2), stride=(1, 2, 2),
                        padding=(0, 0, 0), dilation=(1, 1, 1), bias=True, tcp=True):
        super().__init__()
        self.tcp = tcp
        self.convTranspose2d = nn.ConvTranspose2d(in_chans, chans, kernel_size[1:], stride[1:], padding[1:], dilation=dilation[1:], bias=bias)
        if self.tcp:
            self.convTranspose1d = nn.Conv1d(chans, chans, kernel_size[0], stride[0], padding[0], dilation=dilation[0], bias=bias)
        else:
            self.convTranspose1d = nn.Conv1d(chans, chans, kernel_size[0], stride[0], 1, dilation=dilation[0], bias=bias)
    
    def forward(self, x):
        b=1
        t, c, d1, d2 = x.size()
        x = x.view(b*t, c, d1, d2)
        x = F.relu(self.convTranspose2d(x))
        
        c, dr1, dr2 = x.size(1), x.size(2), x.size(3)
        x = x.view(b, t, c, dr1, dr2)
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(b*dr1*dr2, c, t)
        if self.tcp:
            x = torch.cat((x[:,:,t-1:t], x, x[:,:,0:1]), dim=-1)
        x = self.convTranspose1d(x)

        #Final output
        out_c, out_t = x.size(1), x.size(2)
        x = x.view(b, dr1, dr2, out_c, out_t)
        x = x.permute(0, 4, 3, 1, 2).contiguous().reshape(-1, out_c, dr1, dr2)
        return x
    
class PromptBlock(nn.Module):
    def __init__(self, dim, prompt_size, learnable_input_prompt = True):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(1, dim, prompt_size, prompt_size), requires_grad=learnable_input_prompt)  
        self.conv1 = Conv2_1d(dim*2, dim, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1))
        self.conv2 = Conv2_1d(dim*2, dim, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1))
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        prompt_resized = F.interpolate(self.prompt, (H, W), mode="bilinear")

        x_p = torch.cat((x, prompt_resized.expand(B, -1, -1, -1)), dim=1)
        x_prompt = self.conv1(x_p)+self.conv(x)
        prompt_embed = self.conv2(x_p)
        prompt_embed = F.adaptive_avg_pool2d(prompt_embed, output_size=(1,1)).squeeze(-1).squeeze(-1)
        prompt_embed = torch.mean(prompt_embed, dim=0, keepdim=True)
        return x_prompt, prompt_embed

class TA_SE_Block_New(nn.Module):
    def __init__(self, chans):
        super().__init__()
        self.chans = chans
        self.conv_x =  Conv2_1d(self.chans, self.chans, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1))
        self.conv_h =  Conv2_1d(self.chans, self.chans, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(chans, chans, 1)

        self.relu = nn.LeakyReLU(inplace=True)
        
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Sequential(
            nn.Linear(chans, chans*2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(chans*2, chans, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x, hh):
        t, ch, h, w = x.shape
        
        x1 = self.conv_x(x)
        h1 = self.conv_h(hh)
        y = self.relu(x1+h1)

        y_1 = self.global_avg_pool(y).squeeze()
        y_2 = y_1.permute(1,0)

        corrs = torch.mm(y_1, y_2)/(self.chans**0.5)
        corrs = self.softmax(corrs+1e-6)

        v = y.reshape(t, -1)
        out = torch.mm(corrs, v).reshape(t, self.chans, h, w)  # here try plusing y
        
        out_1 = self.global_avg_pool(out).squeeze()
        c_out = self.fc(out_1).view(t, self.chans, 1, 1)
        f_out = out*c_out.expand_as(out)+self.conv(y)
        return f_out

class Mlp(nn.Module):
    def __init__(self, in_dim, hid_dim=None, out_dim=None, act=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hid_dim = hid_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = act()
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x): 
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class FliM_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w_lin = nn.Linear(dim*2, dim)
        self.b_lin = nn.Linear(dim*2, dim)
    
    def forward(self, x, z):
        bt = x.shape[0]
        x_c = F.adaptive_avg_pool2d(x, output_size=(1,1)).squeeze(-1).squeeze(-1)
        z = torch.cat((x_c, z.repeat(bt, 1)), dim=-1)
        w_z = self.w_lin(z).unsqueeze(-1).unsqueeze(-1)
        b_z = self.b_lin(z).unsqueeze(-1).unsqueeze(-1)
        z_info = w_z+b_z
        z_info = torch.mean(z_info, dim=0, keepdim=True)
        return x+w_z*x+b_z, z_info.squeeze(-1).squeeze(-1)
    
class UNet_Block_Prompt_New(nn.Module):
    def __init__(self, chans=64):
        super().__init__()
        self.chans = chans
        self.chans1 = chans

        self.prompt1 = PromptBlock(dim=chans, prompt_size=64)
        self.prompt2 = PromptBlock(dim=chans, prompt_size=32)

        self.flim1 = FliM_Block(self.chans)
        self.flim2 = FliM_Block(self.chans)
        self.flim3 = FliM_Block(self.chans)
        self.flim4 = FliM_Block(self.chans)
        self.flim5 = FliM_Block(self.chans)
        
        self.d_ts1 = TA_SE_Block_New(self.chans)
        self.d_conv1 = Conv2_1d(self.chans, self.chans)
        self.d_ts2 = TA_SE_Block_New(self.chans)
        self.d_conv2 = Conv2_1d(self.chans, self.chans1, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1))
        
        self.bn = TA_SE_Block_New(self.chans1)
        self.d_conv3 = Conv2_1d(self.chans1, self.chans1, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1))

        self.u_conv2 = Conv2_1d(self.chans1, self.chans, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1))
        self.u_ts2 = TA_SE_Block_New(self.chans)
        self.u_conv1 = ConvTranspose2_1d(self.chans, self.chans)
        self.u_ts1 = TA_SE_Block_New(self.chans)
        
        self.conv11 = nn.Conv2d(self.chans, self.chans1, 1)
        self.conv21 = nn.Conv2d(self.chans1, self.chans, 1)
        self.conv31 = nn.Conv2d(self.chans1, self.chans1, 1)
        self.conv41 = nn.Conv2d(self.chans1, self.chans1, 1)
        self.conv51 = nn.Conv2d(self.chans1, self.chans1, 1)
        
        self.conv2 = Conv2_1d(self.chans*2, self.chans, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1))
        self.conv1 = Conv2_1d(self.chans*2, self.chans, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1))
        
        self.conv =  nn.Conv2d(self.chans, 2, 5, padding = 2)
        self.conv0 = nn.Conv2d(2, self.chans, 3, padding = 3//2)

        self.relu = nn.LeakyReLU(inplace=True)

    def sens_expand(self, x, sens_maps):
        """
        Forward operator: from coil-combined image-space to k-space.
        """
        return utils.fft2c(utils.complex_mul(x, sens_maps))

    def sens_reduce(self, x, sens_maps):
        """
        Backward operator: from k-space to coil-combined image-space.
        """
        x = utils.ifft2c(x)
        return utils.complex_mul(x, utils.complex_conj(sens_maps)).sum(
            dim=2, keepdim=True,
        )

    def norm(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)
        return (x - mean) / (std+1e-6), mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean
        

    def data_consistency(self, img, k0, mask, sens_maps, noise_lvl=None):
        v = noise_lvl
        k = torch.view_as_complex(self.sens_expand(img, sens_maps))
        mask = mask[:,:,None,:,:]
        if v is not None:  # noisy case
            out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
        else:  # noiseless case
            out = (1 - mask) * k + mask * k0
        out = self.sens_reduce(torch.view_as_real(out), sens_maps).squeeze(2)  # b t cm h w 2
        return out
    
    def forward(self, ref_kspace, x, net, mask, sens_maps, prompt_embed): 
        b, _, h, w, _ = x.shape
        prompts = []
        prompts_img = []
        x = x.permute(0,1,4,2,3).contiguous()
        x = x.reshape(b, -1, h, w)
        x, mean, std = self.norm(x)
        x = x.reshape(b, 2, -1, h, w)

        x = x.permute(2,0,1,3,4).contiguous()
        t, b, _, h, w = x.shape
        x = x.float().contiguous()    
        x0 = self.conv0(x.reshape(-1, 2, h, w))

        x00, prompt_tmp = self.flim1(x0, prompt_embed)
        net['x0'] = self.d_ts1(x00, net['x0'])
        net['x0'] = net['x0']+self.conv11(x0)
        x1 = self.d_conv1(net['x0'])
        prompts.append(prompt_tmp)

        x01, prompt_tmp = self.flim2(x1, prompt_embed)
        net['x1'] = self.d_ts2(x01, net['x1'])
        net['x1'] = net['x1']+self.conv21(x1)
        x2 = self.d_conv2(net['x1'])
        prompts.append(prompt_tmp)

        x02, prompt_tmp = self.flim3(x2, prompt_embed)
        net['x2'] = self.bn(x02, net['x2'])
        net['x2'] = net['x2']+self.conv31(x2)
        x2 = self.d_conv3(net['x2'])
        prompts.append(prompt_tmp)

        x3 = self.u_conv2(x2)
        x03, prompt_tmp = self.flim4(x3, prompt_embed)
        net['x3'] = self.u_ts2(x03, net['x3'])
        net['x3'], prompt_x3 = self.prompt2(net['x3'])
        net['x3'] = net['x3']+self.conv41(x3)
        net['x3'] = torch.cat((net['x3'], net['x1']), dim=1)
        net['x3'] = self.conv2(net['x3'])
        net['x1'] = net['x3']
        prompts.append(prompt_tmp)
        prompts_img.append(prompt_x3)
   
        x4 = self.u_conv1(net['x3'])
        x04, prompt_tmp = self.flim5(x4, prompt_embed)
        net['x4'] = self.u_ts1(x04, net['x4'])
        net['x4'], prompt_x4 = self.prompt1(net['x4'])
        net['x4'] = net['x4']+self.conv51(x4)
        net['x4'] = torch.cat((net['x4'], net['x0']), dim=1)
        net['x4'] = self.conv1(net['x4'])
        net['x0'] = net['x4']
        prompts.append(prompt_tmp)
        prompts_img.append(prompt_x4)

        x01 = self.conv(net['x4'])

        x = x.view(-1, 2, h, w)
        x = x + x01 # tb 2 h w

        x = x.view(-1, b, 2, h, w) # t b 2 h w
        x = x.permute(1,0,2,3,4).contiguous()
        x = x.reshape(b, -1, h, w)
        x = self.unnorm(x, mean, std)
        x = x.reshape(b,-1, 2, h, w)
        x = x.permute(0,1,3,4,2).contiguous()

        x = self.data_consistency(x.unsqueeze(2), ref_kspace, mask, sens_maps)
        x = x.permute(0,4,2,3,1).contiguous() # b 2 h w t
        
        return x, net, prompts, prompts_img
    
class Classifier_New(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim*2, dim, dim)
        self.lin_kspace = nn.Linear(dim, 3)
        self.lin_acc = nn.Linear(dim, 6)
    
    def forward(self, x, y):
        x = self.norm(x)
        y = self.norm(y)
        x = torch.mean(x, dim=0, keepdim=True)
        y = torch.mean(y, dim=0, keepdim=True)
        x = torch.cat((x,y), dim=-1)
        x = self.mlp(x)
        return self.lin_kspace(x), self.lin_acc(x)

class Prompts_Block(nn.Module):
    def __init__(self, chans=64, kspace_num = 3, acc_num = 6):
        super().__init__()
        self.kspace_prior = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(kspace_num, chans)))
        self.acc_prior = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(acc_num, chans)))
        self.mlp = Mlp(chans*2, chans*2, chans)
    
    def forward(self, kspace_idx, acc_idx):
        kspace_embed = self.kspace_prior[kspace_idx.long()]
        acc_embed = self.acc_prior[acc_idx.long()]
        prompt_embed = torch.cat((kspace_embed, acc_embed), dim=-1)
        prompt_embed = self.mlp(prompt_embed)
        return prompt_embed
    
class UPCMR(nn.Module):
    def __init__(self, num_cascades=8, chans=64, kspace_num = 3, acc_num = 6):
        super().__init__()
        self.num_cascades = num_cascades
        self.chans = chans
        self.kspace_num = kspace_num
        self.acc_num = acc_num
        self.prompts = Prompts_Block(chans, kspace_num, acc_num)
        self.cascade1 = UNet_Block_Prompt_New(chans=chans)
        self.cascade2 = UNet_Block_Prompt_New(chans=chans)
        self.cascade3 = UNet_Block_Prompt_New(chans=chans)
        self.cascade4 = UNet_Block_Prompt_New(chans=chans)
        self.cascade5 = UNet_Block_Prompt_New(chans=chans)
        self.cascade6 = UNet_Block_Prompt_New(chans=chans)
        self.cascade7 = UNet_Block_Prompt_New(chans=chans)
        self.cascade8 = UNet_Block_Prompt_New(chans=chans)
        self.cs = [self.cascade1, self.cascade2, self.cascade3, self.cascade4, self.cascade5, self.cascade6, self.cascade7, self.cascade8]

        self.cascades = nn.ModuleList(
            self.cs[:num_cascades]
        )
        self.classifier = Classifier_New(chans)
        
    def sens_reduce(self, x, sens_maps):
        x = utils.ifft2c(x)
        return utils.complex_mul(x, utils.complex_conj(sens_maps)).sum(
            dim=2, keepdim=True,
        )
    
    def forward(self, ref_kspace, mask, sens_maps, kspace_idx, acc_idx, train=True):
        kspace_pred = torch.view_as_real(ref_kspace).clone()
        sens_maps = sens_maps.unsqueeze(1)
        x_ref = self.sens_reduce(torch.view_as_real(ref_kspace), sens_maps).squeeze(2) # b t h w 2
        x = x_ref.clone().permute(0,4,2,3,1).contiguous() # b 2 h w t
        b, ch, h, w, t = x.size()
        kspace_pred = kspace_pred.reshape(b, -1, h, w, 2)

        prompt_embed = self.prompts(kspace_idx, acc_idx)

        size_h = [t*b, self.chans, h//2, w//2]
        
        net = {}
        net['x0'] = torch.Tensor(torch.zeros([t*b, self.chans, h, w])).requires_grad_(True).to(x.device)
        net['x4'] = torch.Tensor(torch.zeros([t*b, self.chans, h, w])).requires_grad_(True).to(x.device)
        net['x1'] = torch.Tensor(torch.zeros(size_h)).requires_grad_(True).to(x.device)
        net['x2'] = torch.Tensor(torch.zeros(size_h)).requires_grad_(True).to(x.device)
        net['x3'] = torch.Tensor(torch.zeros(size_h)).requires_grad_(True).to(x.device)

        if train:
            prompt_embeds = []
            prompt_embeds_img = []
            for cascade in self.cascades:
                x, net, prompts, prompts_img = cascade(ref_kspace, x, net, mask, sens_maps, prompt_embed)
                prompt_embeds.extend(prompts)
                prompt_embeds_img.extend(prompts_img)
            prompt_embeds = torch.cat(prompt_embeds, dim=0)
            prompt_embeds_img = torch.cat(prompt_embeds_img, dim=0)
            ksapce_class, acc_class = self.classifier(prompt_embeds, prompt_embeds_img)
            return x.permute(0,4,2,3,1).contiguous(), ksapce_class, acc_class
        else:
            for cascade in self.cascades:
                x, net, _, _ = cascade(ref_kspace, x, net, mask, sens_maps, prompt_embed)
            return x.permute(0,4,2,3,1).contiguous()

