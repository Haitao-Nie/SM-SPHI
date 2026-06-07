import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np

class Data_Process(object):
    def __init__(self, spatial_keep_ratio=0.3, spatial_mask_seed=42):
        self.noise_sigma = 0
        self.hsi_max = []
        self.spatial_keep_ratio = spatial_keep_ratio
        self.spatial_mask_seed = spatial_mask_seed
        self._spatial_gate_cache = {}

    def _generate_blue_noise_like_gate(self, patch_size):
        cache_key = (patch_size[0], patch_size[1], float(self.spatial_keep_ratio), int(self.spatial_mask_seed))
        if cache_key in self._spatial_gate_cache:
            return self._spatial_gate_cache[cache_key]

        height, width = patch_size
        total = height * width
        keep_count = max(1, min(total, int(round(total * self.spatial_keep_ratio))))

        rng = np.random.default_rng(self.spatial_mask_seed)
        coords = np.stack(np.meshgrid(np.arange(height), np.arange(width), indexing='ij'), axis=-1).reshape(-1, 2)

        # Greedy farthest-point sampling gives a deterministic blue-noise-like pattern:
        # selected points repel each other and spread evenly over the grid.
        first_idx = int(rng.integers(total))
        selected = [first_idx]
        chosen = np.zeros(total, dtype=bool)
        chosen[first_idx] = True

        min_dist2 = np.sum((coords - coords[first_idx]) ** 2, axis=1).astype(np.float64)
        min_dist2[first_idx] = -1.0

        for _ in range(1, keep_count):
            jitter = rng.random(total) * 1e-6
            candidate_scores = min_dist2 + jitter
            candidate_scores[chosen] = -1.0
            next_idx = int(np.argmax(candidate_scores))
            selected.append(next_idx)
            chosen[next_idx] = True
            dist2 = np.sum((coords - coords[next_idx]) ** 2, axis=1).astype(np.float64)
            min_dist2 = np.minimum(min_dist2, dist2)
            min_dist2[chosen] = -1.0

        gate = np.zeros((height, width), dtype=np.float32)
        selected_coords = coords[np.array(selected)]
        gate[selected_coords[:, 0], selected_coords[:, 1]] = 1.0

        self._spatial_gate_cache[cache_key] = gate
        return gate

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
        spatial_gate_np = self._generate_blue_noise_like_gate(patch_size)
        spatial_gate = torch.from_numpy(spatial_gate_np).to(mask_patch.device, dtype=mask_patch.dtype)
        mask_patch = mask_patch * spatial_gate.unsqueeze(0)
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








