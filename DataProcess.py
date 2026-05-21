import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

class Data_Process(object):
    def __init__(self):
        self.noise_sigma = 0
        self.hsi_max = []

    def add_noise(self, inputs, sigma):
        noise = torch.zeros_like(inputs)
        noise.normal_(0, sigma)
        noisy = inputs + noise
        noisy = torch.clamp(noisy, 0, 1.0)
        return noisy

    # Extract one fixed center mask patch and replicate it for a batch
    def get_fixed_center_mask_patches(self, mask, image_size, patch_size, batch_size):
        fixed_h = max((image_size[0] - patch_size[0]) // 2, 0)
        fixed_w = max((image_size[1] - patch_size[1]) // 2, 0)
        mask_patch = mask[:, fixed_h:fixed_h + patch_size[0], fixed_w:fixed_w + patch_size[1]]
        mask_patch = mask_patch / mask_patch.max()
        mask_patches = mask_patch.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return mask_patches
            
        
    #Forward model of snapshot hyperspectral imaging for generating input synthesized measurements from hyperspectral targets
    def get_mos_hsi(self, hsi, mask, sigma=0, mos_size=2048, hsi_input_size=512, hsi_target_size=512, init_div_rat=10):
        if not hsi_input_size == hsi_target_size:
            hsi_out = self.extend_spatial_resolution(hsi, extend_rate=hsi_target_size / hsi_input_size)
        else:
            hsi_out=hsi

        if not mos_size == hsi_input_size:
            hsi_expand = self.extend_spatial_resolution(hsi, extend_rate=mos_size / hsi_input_size)
        else:
            hsi_expand=hsi

        mos = torch.sum(hsi_expand * mask, dim=1).unsqueeze(1)
        mos_max = torch.max(mos.view(mos.shape[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)

        #normalize the input and target data using the adaptive variable
        output_hsi = hsi_out / mos_max * init_div_rat
        input_mos = mos / mos_max


        if isinstance(sigma, tuple):
            select_noise_sigma = sigma[random.randint(0, len(sigma) - 1)]
        else: 
            select_noise_sigma = sigma

        input_mos = self.add_noise(input_mos, select_noise_sigma)

        return input_mos, output_hsi


    def extend_spatial_resolution(self, hsi, extend_rate):
        hsi_extend = torch.nn.functional.interpolate(hsi, recompute_scale_factor=True, scale_factor=extend_rate)
        return hsi_extend








