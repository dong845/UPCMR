import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import h5py
from glob import glob
import random
import models.utils as utils
import torch
from torchvision import transforms
from numpy.fft import fft, fft2, ifftshift, fftshift, ifft2
import matplotlib.pyplot as plt
import math
import re
from tqdm import tqdm


class CineDataset_Task2_Test_F(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.folders = sorted(os.listdir(self.data_root))
        self.files = []
        for folder in self.folders:
            path = os.path.join(self.data_root, folder)
            files = glob(os.path.join(path, "*.h5"))
            self.files.extend(list(files))
        
        self.acc_factors = ["4", "8", "12", "16", "20", "24"]
        self.kspaces = ["Gaussian", "Radial", "Uniform"]
        self.contrast_map = {"cine_lax": 0, 
                        "cine_sax": 1, 
                        "cine_lvot": 2, 
                        "T1map": 3,
                        "T2map": 4,
                        "aorta_sag": 5,
                        "aorta_tra": 6,
                        "tagging": 7}

        self.kspace_map = {
            "Gaussian": 0,
            "Radial": 1,
            "Uniform": 2
        }

        self.acc_map = {
            "4": 0,
            "8": 1,
            "12": 2,
            "16": 3,
            "20": 4,
            "24": 5
        }

    
    def __len__(self):
        return len(self.files)
    
    def ifft2c(self, kspace):
        axes = (-2, -1)
        return fftshift(fft2(ifftshift(kspace, axes=axes), axes=axes, norm='ortho'), axes=axes)

    def fft2c(self, img):
        axes = (-2, -1)
        return fftshift(ifft2(ifftshift(img, axes=axes), axes=axes, norm='ortho'), axes=axes)
    
    def norm(self, kspace):
        img = self.ifft2c(kspace)
        img = img/np.max(np.abs(img))
        kspace = self.fft2c(img)
        return kspace
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        folder_name = file_path.split('/')[-2]
        name = os.path.basename(file_path).split('.')[0]
        name_tmp = name.split('_')[-1]

        for kspace in self.kspaces:
            if kspace in name:
                kspace_val = self.kspace_map[kspace]
                break
        for key in self.contrast_map:
            if key in name:
                contrast_val = self.contrast_map[key]
                break
        pattern = r'\d+'
        acc = str(re.search(pattern, name_tmp).group())
        acc_val = self.acc_map[acc]

        with h5py.File(file_path, 'r') as ff:
            kt_mask = np.array(ff["Mask"])
            under_kspace = np.array(ff["UnderSample"])
            csmap = np.array(ff["CSM"])
        
        under_kspace = self.norm(np.transpose(under_kspace, (3, 4, 2, 0, 1)))
        kt_mask = np.transpose(kt_mask, (2,0,1))
            
        name = f"{name}"
        if self.transform is not None:
            return self.transform(under_kspace), kt_mask, csmap, name, folder_name, contrast_val, kspace_val, acc_val
        return under_kspace, kt_mask, csmap, name, folder_name, contrast_val, kspace_val, acc_val


class CineDataset_Task2_Sequential_F1(Dataset):
    def __init__(self, data_root, acc_factors, weights, kspace, k_weights, mode="train",transform=None, slice_fix=False):
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.folders = sorted(os.listdir(self.data_root))
        idex = len(self.folders)-2
        if self.mode=="train":
            self.folders = self.folders[:idex]
        elif self.mode=="val":
            self.folders = self.folders[idex:]
        self.files = []
        for folder in self.folders:
            path = os.path.join(self.data_root, folder)
            files = glob(os.path.join(path, "*.h5"))
            self.files.extend(list(files))
        
        self.acc_factors = acc_factors
        self.weights = weights
        self.k_weights = k_weights
        self.kspace = kspace
        self.slice_fix = slice_fix
        self.kspaces = ["Gaussian", "Radial", "Uniform"]
        self.contrast_map = {"cine_lax": 0, 
                        "cine_sax": 1, 
                        "cine_lvot": 2, 
                        "T1map": 3,
                        "T2map": 4,
                        "aorta_sag": 5,
                        "aorta_tra": 6,
                        "tagging": 7}

        self.kspace_map = {
            "Gaussian": 0,
            "Radial": 1,
            "Uniform": 2
        }

        self.acc_map = {
            "4": 0,
            "8": 1,
            "12": 2,
            "16": 3,
            "20": 4,
            "24": 5
        }
    
    def __len__(self):
        return len(self.files)
    
    def ifft2c(self, kspace):
        axes = (-2, -1)
        return fftshift(fft2(ifftshift(kspace, axes=axes), axes=axes, norm='ortho'), axes=axes)

    def fft2c(self, img):
        axes = (-2, -1)
        return fftshift(ifft2(ifftshift(img, axes=axes), axes=axes, norm='ortho'), axes=axes)
    
    def norm(self, kspace):
        img = self.ifft2c(kspace)
        img = img/np.max(np.abs(img))
        kspace = self.fft2c(img)
        return kspace

    def __getitem__(self, idx):
        file_path = self.files[idx]
        if self.mode=="train":
            name = os.path.basename(file_path).split('.')[0]
            self.acc = random.choices(self.acc_factors, weights=self.weights, k=1)[0]
            if self.kspace is None:
                kspace = random.choices(self.kspaces, weights=self.k_weights, k=1)[0]
            else:
                kspace = random.choices(self.kspace, weights=self.k_weights, k=1)[0]
            contrast_val = self.contrast_map[name]
            kspace_val = self.kspace_map[kspace]
            acc_val = self.acc_map[self.acc]

            with h5py.File(file_path, 'r') as ff:
                full_ksapce = np.array(ff["FullSample"])
                slice_num = full_ksapce.shape[-2]
                if self.slice_fix:
                    slice_id = slice_num//2
                else:
                    slice_id = random.choice(list(range(slice_num)))
                full_ksapce_slice = full_ksapce[..., slice_id, :]
                kt_mask = np.array(ff[f"kt{kspace}{self.acc}"])
                under_kspace = full_ksapce_slice*np.expand_dims(kt_mask, axis=2)
                if kspace == "Gaussian" or kspace=="Uniform":
                    csmap = np.array(ff["CSM_GU"])
                else:
                    csmap = np.array(ff["CSM_RA"])
                csmap = csmap[slice_id]
                sense_map = np.array(ff["CSM"])[slice_id]

            full_ksapce_slice = self.norm(np.transpose(full_ksapce_slice, (3, 2, 0, 1)))
            under_kspace = self.norm(np.transpose(under_kspace, (3, 2, 0, 1)))            
            kt_mask = np.transpose(kt_mask, (2,0,1))

            name = f"{name}_kt{kspace}{self.acc}_slice{slice_id+1}"
            if self.transform is not None:
                return self.transform(full_ksapce_slice), self.transform(under_kspace), kt_mask, csmap, sense_map, name, contrast_val, kspace_val, acc_val
            return full_ksapce_slice, under_kspace, kt_mask, csmap, sense_map, name, contrast_val, kspace_val, acc_val
        else:
            return file_path
