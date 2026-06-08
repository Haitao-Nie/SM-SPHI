import argparse
import os
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from colour import MSDS_CMFS, SDS_ILLUMINANTS, XYZ_to_sRGB
from colour.colorimetry import SpectralShape

from architecture import model_generator
from DataProcess import Data_Process
from getdataset import HyperspectralDataset
from my_utils import load_mat_compat


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly visualize one test sample reconstructed by HyperspecI-V1"
    )
    parser.add_argument("--method", type=str, default="V1_srnet_pconv", help="Model name")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
    parser.add_argument("--mask_path", type=str, default="./MASK/Mask_HyperspecI_V1.mat", help="Path to sensing mask")
    parser.add_argument("--checkpoint_root", type=str, default="./exp/HyperspecI_V1/", help="Root folder that stores ratio-specific checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="best_model.pth", help="Checkpoint file name inside the ratio-specific folder")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path to checkpoint; auto-resolved by spatial_keep_ratio if omitted")
    parser.add_argument("--test_data_path", type=str, default="./ICVL_64/test/", help="Path to test patch dataset")
    parser.add_argument("--output_folder", type=str, default="./vis_test_v1/", help="Folder for visualization outputs")
    parser.add_argument("--sample_index", type=int, default=None, help="Fixed test sample index; random if omitted")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed for random sample selection")
    parser.add_argument("--sigma", type=float, nargs="+", default=(0, 1 / 255, 2 / 255, 3 / 255), help="Gaussian noise sigma used in synthesized measurement")
    parser.add_argument("--start_dir", type=int, nargs=2, default=(0, 0), help="Top-left crop position in the calibrated mask")
    parser.add_argument("--image_size", type=int, nargs=2, default=(2048, 2048), help="Full calibrated mask size")
    parser.add_argument("--patch_size", type=int, nargs=2, default=(64, 64), help="Patch size")
    parser.add_argument("--spatial_keep_ratio", type=float, default=0.3, help="Fraction of spatial locations kept in binary spatial gate")
    parser.add_argument("--spatial_mask_seed", type=int, default=42, help="Seed for deterministic spatial gate generation")
    parser.add_argument("--spatial_mask_cache_root", type=str, default="./MASK/blue_noise_masks", help="Cache directory for deterministic spatial masks")
    parser.add_argument("--spectrum_xy", type=int, nargs=2, default=None, help="Spectrum probe location as (x, y); default uses patch center")
    parser.add_argument("--hist_bins", type=int, default=50, help="Number of bins for absolute-error histogram")
    return parser.parse_args()


def build_ratio_tag(spatial_keep_ratio):
    return f"ratio_{spatial_keep_ratio:.2f}"


def resolve_pretrained_model_path(opt):
    if opt.pretrained_model_path is not None:
        return opt.pretrained_model_path
    return os.path.join(opt.checkpoint_root, build_ratio_tag(opt.spatial_keep_ratio), opt.checkpoint_name)


def resolve_output_folder(opt):
    return os.path.join(opt.output_folder, build_ratio_tag(opt.spatial_keep_ratio))


def load_mask(mask_path, start_dir, image_size):
    mask_init = load_mat_compat(mask_path)["mask"]
    if mask_init.ndim == 3 and mask_init.shape[0] > mask_init.shape[-1]:
        mask_init = np.transpose(mask_init, (2, 0, 1))
    mask = mask_init[
        :,
        start_dir[0]:start_dir[0] + image_size[0],
        start_dir[1]:start_dir[1] + image_size[1],
    ]
    mask = np.maximum(mask, 0)
    mask = mask / np.maximum(mask.max(), 1e-8)
    return torch.from_numpy(mask.astype(np.float32)).cuda()


def tensor_to_numpy_chw(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32)


def compute_error_maps(gt_hsi, recon_hsi):
    abs_err = np.abs(recon_hsi - gt_hsi)
    mae_map = abs_err.mean(axis=0)
    return abs_err, mae_map


def resolve_spectrum_probe(hsi_cube, spectrum_xy):
    _, height, width = hsi_cube.shape
    if spectrum_xy is None:
        x = width // 2
        y = height // 2
    else:
        x = int(np.clip(spectrum_xy[0], 0, width - 1))
        y = int(np.clip(spectrum_xy[1], 0, height - 1))
    return x, y


def save_reconstruction_h5(save_path, mos, gt_hsi, recon_hsi, wavelengths_nm, sample_name):
    with h5py.File(save_path, "w") as f:
        f["sample_name"] = np.bytes_(sample_name)
        f["mos"] = mos
        f["gt_hsi"] = gt_hsi
        f["recon_hsi"] = recon_hsi
        f["wavelengths_nm"] = wavelengths_nm


def build_colour_tables(wavelengths_nm):
    interval = int(round(float(wavelengths_nm[1] - wavelengths_nm[0]))) if len(wavelengths_nm) > 1 else 1
    shape = SpectralShape(int(round(float(wavelengths_nm[0]))), int(round(float(wavelengths_nm[-1]))), interval)

    cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape)
    illuminant = SDS_ILLUMINANTS["D65"].copy().align(shape)

    cmf_values = cmfs.values.astype(np.float32)
    illuminant_values = illuminant.values.astype(np.float32)
    delta = float(shape.interval)
    k = 100.0 / np.maximum(np.sum(cmf_values[:, 1] * illuminant_values) * delta, 1e-8)

    return cmf_values, illuminant_values, delta, k


def hsi_to_srgb_with_colour(hsi_cube, colour_tables, scale=None):
    cmf_values, illuminant_values, delta, k = colour_tables
    weighted = hsi_cube * illuminant_values[:, None, None]
    xyz = np.tensordot(cmf_values.T, weighted, axes=(1, 0)) * (k * delta)
    xyz = np.transpose(xyz, (1, 2, 0)).astype(np.float32)
    xyz = xyz / 100.0
    rgb = XYZ_to_sRGB(xyz)
    rgb = np.clip(rgb, 0.0, None)
    if scale is None:
        scale = float(np.percentile(rgb, 99.5))
    rgb = rgb / max(scale, 1e-8)
    return np.clip(rgb, 0.0, 1.0), scale


def main():
    opt = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    cudnn.benchmark = True

    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)

    pretrained_model_path = resolve_pretrained_model_path(opt)
    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_model_path}")

    output_folder = resolve_output_folder(opt)
    os.makedirs(output_folder, exist_ok=True)
    recon_h5_folder = os.path.join(output_folder, "recon_h5")
    os.makedirs(recon_h5_folder, exist_ok=True)

    dataset = HyperspectralDataset(root_dir=opt.test_data_path)
    if opt.sample_index is None:
        sample_index = random.randrange(len(dataset))
    else:
        sample_index = int(np.clip(opt.sample_index, 0, len(dataset) - 1))

    sample_name = dataset.files[sample_index]
    print(f"checkpoint: {pretrained_model_path}")
    print(f"Selected sample [{sample_index}]: {sample_name}")

    gt_hsi = dataset[sample_index].unsqueeze(0).cuda()
    mask = load_mask(opt.mask_path, opt.start_dir, opt.image_size)

    data_processing = Data_Process(
        spatial_keep_ratio=opt.spatial_keep_ratio,
        spatial_mask_seed=opt.spatial_mask_seed,
        spatial_mask_cache_root=opt.spatial_mask_cache_root,
    )
    mask_patch = data_processing.get_fixed_center_mask_patches(
        mask=mask,
        image_size=opt.image_size,
        patch_size=opt.patch_size,
        batch_size=1,
    )
    mos, gt_target = data_processing.get_mos_hsi(
        hsi=gt_hsi,
        mask=mask_patch,
        sigma=tuple(opt.sigma),
        mos_size=opt.patch_size[0],
        hsi_input_size=opt.patch_size[0],
        hsi_target_size=opt.patch_size[0],
    )

    model = model_generator(opt.method, pretrained_model_path)
    model.eval()

    with torch.no_grad():
        recon_hsi = model(mos, mask_patch)
        recon_hsi = torch.clamp(recon_hsi, min=0.0)

    mos_np = tensor_to_numpy_chw(mos[0])[0]
    gt_np = tensor_to_numpy_chw(gt_target[0])
    recon_np = tensor_to_numpy_chw(recon_hsi[0])
    wavelengths_nm = np.linspace(400.0, 700.0, gt_np.shape[0], dtype=np.float32)

    colour_tables = build_colour_tables(wavelengths_nm)
    gt_rgb_raw, gt_scale = hsi_to_srgb_with_colour(gt_np, colour_tables, scale=None)
    recon_rgb_raw, recon_scale = hsi_to_srgb_with_colour(recon_np, colour_tables, scale=None)
    shared_scale = max(gt_scale, recon_scale, 1e-8)
    gt_rgb, _ = hsi_to_srgb_with_colour(gt_np, colour_tables, scale=shared_scale)
    recon_rgb, _ = hsi_to_srgb_with_colour(recon_np, colour_tables, scale=shared_scale)

    abs_err, mae_map = compute_error_maps(gt_np, recon_np)
    probe_x, probe_y = resolve_spectrum_probe(gt_np, opt.spectrum_xy)
    gt_spectrum = gt_np[:, probe_y, probe_x]
    recon_spectrum = recon_np[:, probe_y, probe_x]
    flat_abs_err = abs_err.reshape(-1)

    save_stem = os.path.splitext(sample_name)[0]
    h5_path = os.path.join(recon_h5_folder, f"{save_stem}_recon.h5")
    save_reconstruction_h5(
        save_path=h5_path,
        mos=mos_np,
        gt_hsi=gt_np,
        recon_hsi=recon_np,
        wavelengths_nm=wavelengths_nm,
        sample_name=sample_name,
    )

    vis_path = os.path.join(output_folder, f"{save_stem}_vis.png")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(gt_rgb)
    axes[0, 0].set_title("GT pseudo-RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mos_np, cmap="gray")
    axes[0, 1].set_title("Simulated MOS")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(recon_rgb)
    axes[0, 2].set_title("Reconstructed pseudo-RGB")
    axes[0, 2].axis("off")

    im = axes[1, 0].imshow(mae_map, cmap="magma")
    axes[1, 0].set_title("Mean Absolute Error Map")
    axes[1, 0].axis("off")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].plot(wavelengths_nm, gt_spectrum, label="GT", linewidth=2)
    axes[1, 1].plot(wavelengths_nm, recon_spectrum, label="Recon", linewidth=2)
    axes[1, 1].set_title(f"Spectrum @ ({probe_x}, {probe_y})")
    axes[1, 1].set_xlabel("Wavelength (nm)")
    axes[1, 1].set_ylabel("Reflectance")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[1, 2].hist(flat_abs_err, bins=opt.hist_bins)
    axes[1, 2].set_title("|Recon - GT| histogram")
    axes[1, 2].set_xlabel("Absolute error")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].grid(True, alpha=0.2)

    fig.suptitle(
        f"{save_stem}\nPseudo-RGB rendered with colour / CIE 1931 + D65",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(vis_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Output folder: {output_folder}")
    print(f"Saved visualization to: {vis_path}")
    print(f"Saved reconstruction h5 to: {h5_path}")
    print(f"Shared RGB scale: {shared_scale:.6f}")
    print(f"Spectrum probe: (x={probe_x}, y={probe_y})")


if __name__ == "__main__":
    main()
