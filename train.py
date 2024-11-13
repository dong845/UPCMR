import time
import h5py
import gc
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import models.utils as utils
from cine_dataset import CineDataset_Task2_Test_F, CineDataset_Task2_Sequential_F1
import matplotlib.pyplot as plt
from metrics import calmetric
from warmup_scheduler import GradualWarmupScheduler
from models.UPCMR_Cur import UPCMR
import os
import random 
import numpy as np
import argparse
import wandb
from losses import SSIMLoss
import torch.nn.functional as F
from tqdm import tqdm
from numpy.fft import fft, fft2, ifftshift, fftshift, ifft2
import math
from torchvision import transforms
import datetime

models_name = {
    "UPCMR": UPCMR,
}

acc_factors = ["4", "8", "12", "16", "20", "24"]
kspaces = ["Gaussian", "Radial", "Uniform"]

contrast_map = {"cine_lax": 0, 
    "cine_sax": 1, 
    "cine_lvot": 2, 
    "T1map": 3,
    "T2map": 4,
    "aorta_sag": 5,
    "aorta_tra": 6,
    "tagging": 7}

kspace_map = {
    "Gaussian": 0,
    "Radial": 1,
    "Uniform": 2
}

acc_map = {
    "4": 0,
    "8": 1,
    "12": 2,
    "16": 3,
    "20": 4,
    "24": 5
}

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
setup_seed(42)

def clip_images(image):
    for i in range(image.shape[0]):
        img = np.abs(image[i])
        lower_bound = np.percentile(img, 1)
        upper_bound = np.percentile(img, 99)
        clipped_image = np.clip(img, lower_bound, upper_bound)
        image[i] = (clipped_image - lower_bound) / (upper_bound - lower_bound)
    return image

def visualize(img, path):
    img = clip_images(img)
    hh = img.shape[1]
    img = img[:,hh//4:hh*3//4,:]
    plt.imshow(np.abs(img[0]), cmap="gray")
    plt.axis('off')
    plt.savefig(path)
    plt.close()

def sens_reduce(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    x = utils.ifft2c(x)
    sens_maps = sens_maps[:,None,...]
    return utils.complex_mul(x, utils.complex_conj(sens_maps)).sum(
        dim=2, keepdim=True,
    )

def build_loader(args, mode, acc_factors, weights, kspace, k_weights, slice_fix=False):
    if mode!="test":
        dataset = CineDataset_Task2_Sequential_F1(args.data_root, acc_factors=acc_factors, weights=weights, kspace=kspace, k_weights=k_weights, mode=mode, slice_fix=slice_fix)
    else:
        dataset = CineDataset_Task2_Test_F(args.data_root)
    if mode=="train":
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader

def check_path(folder_path, model_name, tv, fname):
    model_dir = os.path.join(folder_path, model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if tv is None:
        tv_dir = model_dir
    else:
        tv_dir = os.path.join(model_dir, tv)
        if not os.path.exists(tv_dir):
            os.mkdir(tv_dir)
    
    if fname is not None:
        if fname=="gnd" or fname=="und":
            file_dir = os.path.join(tv_dir, fname)
        else:
            file_dir = os.path.join(tv_dir, f"rec_{fname}")
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        return file_dir
    else:
        return tv_dir

def process(acc, kspace, name, file_path):
    def ifft2c(kspace):
        axes = (-2, -1)
        return fftshift(fft2(ifftshift(kspace, axes=axes), axes=axes, norm='ortho'), axes=axes)

    def fft2c(img):
        axes = (-2, -1)
        return fftshift(ifft2(ifftshift(img, axes=axes), axes=axes, norm='ortho'), axes=axes)
    
    def norm(kspace):
        img = ifft2c(kspace)
        img = img/np.max(np.abs(img))
        kspace = fft2c(img)
        return kspace

    with h5py.File(file_path, 'r') as ff:
        full_ksapce = np.array(ff["FullSample"])
        kt_mask = np.array(ff[f"kt{kspace}{acc}"])
        under_kspace = full_ksapce*np.expand_dims(np.expand_dims(kt_mask, axis=2), axis=2)
        if kspace == "Gaussian" or kspace=="Uniform":
            csmap = np.array(ff["CSM_GU"])
        else:
            csmap = np.array(ff["CSM_RA"])
        sense_map = np.array(ff["CSM"])

        full_ksapce = norm(np.transpose(full_ksapce, (3, 4, 2, 0, 1)))
        under_kspace = norm(np.transpose(under_kspace, (3, 4, 2, 0, 1)))
        kt_mask = np.transpose(kt_mask, (2,0,1))

    contrast_idx = torch.tensor(contrast_map[name])
    kspace_idx = torch.tensor(kspace_map[kspace])
    acc_idx = torch.tensor(acc_map[acc])
    return torch.tensor(full_ksapce), torch.tensor(under_kspace), torch.tensor(csmap), torch.tensor(sense_map).unsqueeze(0), torch.tensor(kt_mask), contrast_idx, kspace_idx, acc_idx

def frames_split(j, und_kspace, full_kspace, gnd_img, und_img, mask):
    gnd_itmp = None
    full_ktmp = None
    if j<2:
        und_ktmp = torch.cat((und_kspace[:,j+und_kspace.shape[1]-2:und_kspace.shape[1],...], und_kspace[:,0:j+3,...]), dim=1)
        if full_kspace is not None:
            full_ktmp = torch.cat((full_kspace[:,j+und_kspace.shape[1]-2:und_kspace.shape[1],...], full_kspace[:,0:j+3,...]), dim=1)
        if gnd_img is not None:
            gnd_itmp = torch.cat((gnd_img[:,j+und_kspace.shape[1]-2:und_kspace.shape[1],...], gnd_img[:,0:j+3,...]), dim=1)
        und_itmp = torch.cat((und_img[:,j+und_kspace.shape[1]-2:und_kspace.shape[1],...], und_img[:,0:j+3,...]), dim=1)
        mask_tmp = torch.cat((mask[:,j+und_kspace.shape[1]-2:und_kspace.shape[1],...], mask[:,0:j+3,...]), dim=1)
    elif j>und_kspace.shape[1]-3:
        und_ktmp = torch.cat((und_kspace[:,j-2:und_kspace.shape[1],...], und_kspace[:,0:j+3-und_kspace.shape[1],...]), dim=1)
        if full_kspace is not None:
            full_ktmp = torch.cat((full_kspace[:,j-2:und_kspace.shape[1],...], full_kspace[:,0:j+3-und_kspace.shape[1],...]), dim=1)
        und_itmp = torch.cat((und_img[:,j-2:und_kspace.shape[1],...], und_img[:,0:j+3-und_kspace.shape[1],...]), dim=1)
        if gnd_img is not None:
            gnd_itmp = torch.cat((gnd_img[:,j-2:und_kspace.shape[1],...], gnd_img[:,0:j+3-und_kspace.shape[1],...]), dim=1)
        mask_tmp = torch.cat((mask[:,j-2:und_kspace.shape[1],...], mask[:,0:j+3-und_kspace.shape[1],...]), dim=1)
    else:
        und_ktmp = und_kspace[:,j-2:j+3,...]
        if full_kspace is not None:
            full_ktmp = full_kspace[:,j-2:j+3,...]
        und_itmp = und_img[:,j-2:j+3,...]
        if gnd_img is not None:
            gnd_itmp = gnd_img[:,j-2:j+3,...]
        mask_tmp = mask[:,j-2:j+3,...]
    return und_ktmp, full_ktmp, und_itmp, gnd_itmp, mask_tmp

def val_infer(model, data_loader):
    model.eval()
    test_loss = 0
    nmses = []
    psnrs = []
    ssims = []
    names = []
    img_unds = []
    img_recs = []
    img_gnds = []
    criterion = torch.nn.L1Loss().cuda()
    acc_factors_new = ["4", "8", "12", "16", "20", "24"]
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
        for i, file_path in enumerate(tqdm(data_loader)):
            file_path = file_path[0]
            folder_name = file_path.split('/')[-2]
            name = os.path.basename(file_path).split('.')[0]
            times = 0
            for acc_factor in acc_factors_new:
                for kspace in kspaces:
                    full_kspace, under_kspace, csmap, sense_map, kt_mask, contrast_idx, kspace_idx, acc_idx = process(acc_factor, kspace, name, file_path)
                    full_kspace = full_kspace.unsqueeze(0).cuda()
                    under_kspace = under_kspace.unsqueeze(0).cuda()
                    csmap = csmap.unsqueeze(0).cuda()
                    kt_mask = kt_mask.unsqueeze(0).cuda()
                    contrast_idx = contrast_idx.cuda()
                    kspace_idx = kspace_idx.cuda()
                    acc_idx = acc_idx.cuda()
                    for j in range(csmap.shape[1]):
                        csm_tmp = csmap[:,j,...].clone().detach()
                        sense_map_tmp = sense_map[:,j,...].clone().detach()
                        full_kspace_tmp = full_kspace[:,j,...].clone().detach()
                        und_kspace_tmp = under_kspace[:,j,...].clone().detach()
                        sense_map_tmp = torch.view_as_real(sense_map_tmp).float().cuda()
                        csm_tmp = torch.view_as_real(csm_tmp).float().cuda()
                        with torch.cuda.amp.autocast():
                            rec_image = model(und_kspace_tmp, kt_mask.float(), csm_tmp, kspace_idx, acc_idx, train=False)
                        full_image = sens_reduce(torch.view_as_real(full_kspace_tmp).cuda(), sense_map_tmp).squeeze(2).cuda()
                        und_image = sens_reduce(torch.view_as_real(und_kspace_tmp).cuda(), sense_map_tmp).squeeze(2).cuda()
                        t_loss = criterion(rec_image, full_image)
                        times+=1
                        test_loss+=t_loss.item()
                        rec_image = torch.view_as_complex(rec_image).squeeze(0).cpu().numpy()
                        gnd_image = torch.view_as_complex(full_image).squeeze(0).cpu().numpy()
                        und_image = torch.view_as_complex(und_image).squeeze(0).cpu().numpy()
                        if j==csmap.shape[1]//2:
                            img_unds.append(und_image)
                            img_recs.append(rec_image)
                            img_gnds.append(gnd_image)
                            names.append(f"{folder_name}_{name}_kt{kspace}{acc_factor}_slice{j+1}")
                        [psnr_array, ssim_array, nmse_array] = calmetric(np.abs(rec_image).transpose(1,2,0), np.abs(gnd_image).transpose(1,2,0))
                        nmses.append(np.mean(nmse_array))
                        psnrs.append(np.mean(psnr_array))
                        ssims.append(np.mean(ssim_array))

    return float(test_loss/times), nmses, psnrs, ssims, names, img_unds, img_recs, img_gnds


def test_infer(model, data_loader):
    model.eval()
    names = []
    img_unds = []
    img_recs = []
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
        for i, (und_kspace, mask, sense_map, name, contrast_idx, kspace_idx, acc_idx) in enumerate(tqdm(data_loader)):
            for j in range(sense_map.shape[1]):
                sense_map_tmp = sense_map[:,j,...].clone().detach()
                und_kspace_tmp = und_kspace[:,j,...].clone().detach()
                sense_map_tmp = torch.view_as_real(sense_map_tmp).float().cuda()
                rec_image = model(und_kspace_tmp.cuda(), mask.float().cuda(), sense_map_tmp, train=False)
                
                und_image = torch.view_as_complex(sens_reduce(torch.view_as_real(und_kspace_tmp).cuda(), sense_map_tmp)).cuda()
                names.append(f"{name[0]}_slice{j+1}")
                rec_img = torch.view_as_complex(rec_image).squeeze(0)
                und_img1 = und_image.squeeze(0).squeeze(1)
                rec_img = rec_img.cpu().numpy()
                und_img1 = und_img1.cpu().numpy()
                img_recs.append(rec_img)
                img_unds.append(und_img1)
        return names, img_recs, img_unds
    

def process_val_test(args, model, data_loader, early_stop, f_name, epoch, best_psnr, best_ssim, times, mode="val"):
    if mode=="val":
        test_loss, nmses, psnrs, ssims, names, img_unds, img_recs, img_gnds = val_infer(model, data_loader)
        c_psnr = np.mean(psnrs)
        c_ssim = np.mean(ssims)
        c_nmse = np.mean(nmses)
        wandb.log({"Test_Loss": float(test_loss), 
                    "Val_NMSE": c_nmse, 
                    "Val_PSNR": c_psnr,
                    "Val_SSIM": c_ssim})
        
        if c_ssim<=best_ssim:                   # early stopping
            times+=1
            if times>=20:
                early_stop = True
        else:
            times = 0
        
        if c_psnr>best_psnr and c_ssim>best_ssim:
            best_psnr = c_psnr
            best_ssim = c_ssim
            weight_name = f"{f_name}.pth"
            weight_rec_path = check_path(args.save_weight_path, args.model_name, None, None)
            torch.save(model.state_dict(), os.path.join(weight_rec_path, weight_name))
            print('model parameters saved at %s' % os.path.join(weight_rec_path, weight_name))
            
            val_path = check_path(args.save_val_path, args.model_name, mode, f_name)
            for i in range(len(names)):
                with h5py.File(os.path.join(val_path, f"{names[i]}.h5"), 'w') as f:
                    f["und"] = img_unds[i]
                    f["rec"] = img_recs[i]
                    f["gnd"] = img_gnds[i]
                    f["mean_nmse"] = c_nmse
                    f["mean_psnr"] = c_psnr
                    f["mean_ssim"] = c_ssim
            print("val psnr:", c_psnr)
            print("val ssim:", c_ssim)
            print("val nmse:", c_nmse)

            und_path = check_path(args.save_pic_path, args.model_name, mode, "und")
            gnd_path = check_path(args.save_pic_path, args.model_name, mode, "gnd")
            rec_path = check_path(args.save_pic_path, args.model_name, mode, f_name)
            for i in range(len(names)):
                if epoch==0:
                    visualize(img_unds[i], os.path.join(und_path, f"{names[i]}.png"))
                    visualize(img_gnds[i], os.path.join(gnd_path, f"{names[i]}.png"))
                visualize(img_recs[i], os.path.join(rec_path, f"{names[i]}.png"))
    elif mode=="test":
        names, img_recs, img_unds = test_infer(model, data_loader)
        val_path = check_path(args.save_val_path, args.model_name, mode, f_name)
        for i in range(len(names)):
            with h5py.File(os.path.join(val_path, f"{names[i]}.h5"), 'w') as f:
                f["und"] = img_unds[i]
                f["rec"] = img_recs[i]
        und_path = check_path(args.save_pic_path, args.model_name, mode, "und")
        rec_path = check_path(args.save_pic_path, args.model_name, mode, f_name)
        for i in range(len(names)):
            visualize(img_unds[i], os.path.join(und_path, f"{names[i]}.png"))
            visualize(img_recs[i], os.path.join(rec_path, f"{names[i]}.png"))
    return early_stop, best_psnr, best_ssim, times

def load_weights(args, model):
    pretrained_weights = torch.load(args.pretrain_weight_path)
    model_state_dict = model.state_dict()
    new_state_dict = {}
    for name, param in pretrained_weights.items():
        if name in model_state_dict:
            new_state_dict[name] = param
        else:
            print(f"Skipping weight for {name}, not found in the new model.")
 
    # Update the model's state dict
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)
    return model

def train(args):
    early_stop = False
    times = 0
    best_psnr = 0
    best_ssim = 0
    lr = args.lr
    num_epoch = args.num_epoch
    weight_decay = args.weight_decay
    warmup_epoch = args.warmup_episodes
    interval = args.interval
    f_name = args.model_name   
    wan_name = f"UPCMR_Cur"
    wandb.init(project=wan_name, name=f"{f_name}")  
    
    weights = [1/6]*6
    model = models_name[args.model_name](num_cascades=8)
    model = load_weights(args, model)
    model = model.cuda()
    wandb.watch(model)
    
    scaler = torch.cuda.amp.GradScaler()
    criterion = torch.nn.L1Loss().cuda()
    criterion0 = torch.nn.MSELoss().cuda()
    criterion1 = SSIMLoss().cuda()
    criterion2 = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=lr,betas=(0.9,0.999),eps=1e-5, weight_decay=weight_decay)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch-warmup_epoch, eta_min=2e-5)  # eta_min from 1e-5 to 1e-6
    scheduler = GradualWarmupScheduler(optimizer,
            multiplier=1,total_epoch=warmup_epoch,
            after_scheduler=scheduler_cosine)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total trainable params: %d' % pytorch_total_params)
    train_loss = 1e4

    for epoch in range(0, num_epoch+1):
        gc.collect()
        torch.cuda.empty_cache()
        t_start = time.time()
        train_err = 0
        train_batches = 0
        model.train()
        train_loader = build_loader(args, mode="train", acc_factors=acc_factors, weights=weights, kspace=["Uniform", "Gaussian", "Radial"], k_weights=[1/3, 1/3, 1/3])

        pbar = tqdm(train_loader)
        for i, (full_kspace, und_kspace, mask, sense_map, csm, _, contrast_idx, kspace_idx, acc_idx) in enumerate(pbar):
            gc.collect()
            torch.cuda.empty_cache()
            sense_map = torch.view_as_real(sense_map).float().cuda()
            csm = torch.view_as_real(csm).float().cuda()
            
            with torch.cuda.amp.autocast():
                contrast_idx = contrast_idx.cuda()
                kspace_idx = kspace_idx.cuda()
                acc_idx = acc_idx.cuda()
                rec_image, kspace_class, acc_class = model(und_kspace.cuda(), mask.float().cuda(), sense_map, kspace_idx, acc_idx)
                gnd_image = sens_reduce(torch.view_as_real(full_kspace).cuda(), csm).squeeze(2).cuda()
                with torch.cuda.amp.autocast(enabled=False):
                    rec_image = rec_image.float()
                    gnd_image = gnd_image.float()
                    kspace_class = kspace_class.float()+1e-6
                    acc_class = acc_class.float()+1e-6
                    rec_kspace = utils.fft2c(rec_image).float()
                    gnd_kspace = utils.fft2c(gnd_image).float()
                    rec_image_abs = utils.complex_abs(rec_image+1e-6).float()
                    gnd_image_abs = utils.complex_abs(gnd_image+1e-6).float()

                    ce_loss = 0.025*(criterion2(kspace_class, kspace_idx.long())+criterion2(acc_class, acc_idx.long()))
                    l1_loss = 0.16*criterion(rec_image, gnd_image)+0.16*criterion(rec_kspace, gnd_kspace)
                    ssim_loss = 0.84*criterion1(rec_image_abs, gnd_image_abs)
                    loss = ce_loss+l1_loss+ssim_loss

                train_err += loss.item()
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_batches += args.batch_size
            pbar.set_description(f"Epoch {epoch}, loss: {loss.item():.4f}")

        t_end = time.time()
        train_err = train_err/train_batches
        scheduler.step()
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch+1}, Current learning rate: {param_group['lr']}")

        weight_name = f"{f_name}.pth"
        weight_rec_path = check_path(args.save_weight_path, args.model_name, None, None)
        torch.save(model.state_dict(), os.path.join(weight_rec_path, weight_name))
        print('model parameters saved at %s' % os.path.join(weight_rec_path, weight_name))
        train_loss = train_err

        print(" Epoch {}/{}".format(epoch + 1, num_epoch))
        print(" time: {}s".format(t_end - t_start))
        print(" training loss:\t\t{:.6f}".format(train_err))
        wandb.log({"Train_Loss": train_err})
        
        if epoch%interval==0 or epoch==args.num_epoch:
            gc.collect()
            torch.cuda.empty_cache()
            val_loader = build_loader(args, mode="val", acc_factors=acc_factors, weights=weights, kspace=["Uniform", "Radial", "Gaussian"], k_weights=[1/3,1/3,1/3])
            print("Validation:")
            early_stop, best_psnr, best_ssim, times = process_val_test(args, model, val_loader, early_stop, f_name, epoch, best_psnr, best_ssim, times)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="UPCMR", help='model name')   # CRNN, CineNet, UNet_3D, CRUNet_TCA, CRUNet_TSA
    parser.add_argument('--data_root', type=str, default="***", help='data root')
    parser.add_argument('--save_pic_path', type=str, default="***", help='save path for images')
    parser.add_argument('--save_val_path', type=str, default="***", help='save path for values')
    parser.add_argument('--save_weight_path', type=str, default="***", help='save path for weights')
    parser.add_argument('--pretrain_weight_path', type=str, default="***", help='pretrained path for weights')
    parser.add_argument('--num_epoch', metavar='int', nargs=1, type=int, default=60, help='number of epochs')  # 600 for LAX, 200 for SAX
    parser.add_argument('--batch_size', metavar='int', nargs=1, type=int, default=1, help='batch size')
    parser.add_argument('--lr', metavar='float', nargs=1, type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=1e-2)  # from 1e-5 to 5e-5
    parser.add_argument('--warmup_episodes', type=int, default=6)
    parser.add_argument('--pretrain', type=bool, default=False)
    args = parser.parse_args()
    
    train(args)
