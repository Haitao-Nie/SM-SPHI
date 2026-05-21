from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch
import random
import os
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate
import torch.nn.functional as F
import h5py

def _load_hsi_array(file_path):
    with h5py.File(file_path, 'r') as f:
        if 'hsi' in f:
            arr = f['hsi'][:]
        elif 'patch' in f:
            arr = f['patch'][:]
        elif 'cube' in f:
            arr = f['cube'][:]
        else:
            raise KeyError(f'No supported key in {file_path}. Expected one of: hsi, patch, cube.')
    return np.asarray(arr)

def _to_chw(arr):
    if arr.ndim != 3:
        raise ValueError(f'Expected 3D array, got shape={arr.shape}')
    # If channel is already first (common: C,H,W), keep it.
    # Otherwise treat as H,W,C and transpose to C,H,W.
    if arr.shape[0] <= 200 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        return arr
    return np.transpose(arr, (2, 0, 1))

def _random_crop_chw(hsi, patch_size):
    patch_size_h, patch_size_w = patch_size
    h, w = hsi.shape[1], hsi.shape[2]
    if h < patch_size_h or w < patch_size_w:
        raise ValueError(f'Patch size {patch_size} is larger than sample spatial size {(h, w)}')
    max_h = h - patch_size_h
    max_w = w - patch_size_w
    random_h = random.randint(0, max_h) if max_h > 0 else 0
    random_w = random.randint(0, max_w) if max_w > 0 else 0
    return hsi[:, random_h:random_h + patch_size_h, random_w:random_w + patch_size_w]

def _normalize_patch(x):
    x = x.astype(np.float32)
    vmax = float(np.max(x))
    if vmax > 0:
        x = x / vmax
    return x


class TrainDataset_V1(Dataset):
    def __init__(self, data_path, patch_size, arg=False):

        self.arg = arg
        self.data_path = data_path
        self.patch_size = patch_size

        data_list = os.listdir(data_path)
        data_list.sort()

        self.data_list = data_list
        self.img_num = len(self.data_list)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.data_list[idx])
        hsi = _load_hsi_array(file_path)
        hsi = _to_chw(hsi)

        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)

        output_hsi = _random_crop_chw(hsi, self.patch_size)
        output_hsi = _normalize_patch(output_hsi)

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num

class ValidDataset_V1(Dataset):
    def __init__(self, data_path, patch_size, arg=False):

        self.arg = arg
        self.data_paths = []
        self.patch_size = patch_size

        data_list = os.listdir(data_path)
        data_list.sort()
        for i in range(len(data_list)):

            self.data_paths.append(data_path + data_list[i])

        self.img_num = len(self.data_paths)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        hsi = _load_hsi_array(self.data_paths[idx])
        hsi = _to_chw(hsi)

       
        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)

        output_hsi = _random_crop_chw(hsi, self.patch_size)
        output_hsi = _normalize_patch(output_hsi)

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num






class TrainDataset_V2(Dataset):
    def __init__(self, data_path, patch_size, arg=False):

        self.arg = arg
        self.data_path = data_path
        self.patch_size = patch_size
        self.select_index = np.concatenate((np.arange(0,61,1), np.arange(62, 132, 2)))
        data_list = os.listdir(data_path)
        data_list.sort()

        self.data_list = data_list
        self.img_num = len(self.data_list)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.data_list[idx])
        hsi = _load_hsi_array(file_path)
        hsi = _to_chw(hsi)
        if hsi.shape[0] <= np.max(self.select_index):
            raise ValueError(
                f'V2 dataset expects at least {np.max(self.select_index) + 1} channels before band selection, '
                f'but got {hsi.shape[0]} channels from {file_path}.'
            )
        hsi = hsi[self.select_index, :, :]


        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)

        output_hsi = _random_crop_chw(hsi, self.patch_size)
        output_hsi = _normalize_patch(output_hsi)

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num

class ValidDataset_V2(Dataset):
    def __init__(self, data_path, patch_size, arg=False):

        self.arg = arg
        self.data_paths = []
        self.patch_size = patch_size
        
        self.select_index = np.concatenate((np.arange(0,61,1), np.arange(62, 132, 2)))

        data_list = os.listdir(data_path)
        data_list.sort()
        for i in range(len(data_list)):

            self.data_paths.append(data_path + data_list[i])

        self.img_num = len(self.data_paths)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        hsi = _load_hsi_array(self.data_paths[idx])
        hsi = _to_chw(hsi)
        if hsi.shape[0] <= np.max(self.select_index):
            raise ValueError(
                f'V2 dataset expects at least {np.max(self.select_index) + 1} channels before band selection, '
                f'but got {hsi.shape[0]} channels from {self.data_paths[idx]}.'
            )
        hsi = hsi[self.select_index, :, :]

       
        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)

        output_hsi = _random_crop_chw(hsi, self.patch_size)
        output_hsi = _normalize_patch(output_hsi)

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num





class TestDataset_MOS(Dataset):
    def __init__(self, data_path, data_list, start_dir, image_size, arg=False):

        self.arg = arg
        self.data_path = data_path

        self.start_dir = start_dir
        self.image_size = image_size

        self.data_list = data_list

        self.MOS_list = []

        for i in range(len(data_list)):

            bmp = cv2.imread(self.data_path + self.data_list[i])[:, :, 0]
            bmp = bmp[self.start_dir[0]:self.start_dir[0]+self.image_size[0], self.start_dir[1]:self.start_dir[1] + self.image_size[1]]
            bmp = bmp / bmp.max()
            bmp = bmp.astype(np.float32)
            mos = np.expand_dims(bmp, axis=0)
            self.MOS_list.append(mos)
            
        self.img_num = len(self.data_list)

    def __getitem__(self, idx):
        mos_name = self.data_list[idx]
        mos = self.MOS_list[idx]

        return np.ascontiguousarray(mos), mos_name

    def __len__(self):
        return self.img_num

