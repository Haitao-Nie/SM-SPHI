import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataProcess import Data_Process
from architecture import model_generator
from getdataset import HyperspectralDataset
from my_utils import AverageMeter, Loss_PSNR, Loss_SAM, Loss_SSIM, load_mat_compat


parser = argparse.ArgumentParser(description="Evaluate HyperspecI-V1 on the test set")
parser.add_argument("--method", type=str, default="V1_srnet_pconv", help="Model")
parser.add_argument("--gpu_id", type=str, default="0", help="select gpu")
parser.add_argument("--batch_size", type=int, default=1, help="testing batch size")
parser.add_argument("--mask_path", type=str, default="./MASK/Mask_HyperspecI_V1.mat", help="path of calibrated sensing matrix")
parser.add_argument("--checkpoint_root", type=str, default="./exp/HyperspecI_V1/", help="root folder that stores ratio-specific checkpoints")
parser.add_argument("--checkpoint_name", type=str, default="best_model.pth", help="checkpoint file name inside the ratio-specific folder")
parser.add_argument("--pretrained_model_path", type=str, default=None, help="path to checkpoint; auto-resolved by spatial_keep_ratio if omitted")
parser.add_argument("--test_data_path", type=str, default="./ICVL_64/test/", help="path to test dataset")
parser.add_argument("--sigma", type=float, nargs="+", default=(0, 1 / 255, 2 / 255, 3 / 255), help="Sigma of Gaussian Noise")
parser.add_argument("--start_dir", type=int, nargs=2, default=(0, 0), help="top-left coordinate of the cropped mask region")
parser.add_argument("--image_size", type=int, nargs=2, default=(2048, 2048), help="size of image region used from mask")
parser.add_argument("--patch_size", type=int, nargs=2, default=(64, 64), help="HSI patch size")
parser.add_argument("--spatial_keep_ratio", type=float, default=0.3, help="fraction of spatial locations kept in the binary spatial gate")
parser.add_argument("--spatial_mask_seed", type=int, default=42, help="seed for deterministic spatial gate generation")
parser.add_argument("--spatial_mask_cache_root", type=str, default="./MASK/blue_noise_masks", help="folder used to load/save deterministic blue-noise-like spatial masks")


opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

criterion_psnr = Loss_PSNR()
criterion_ssim = Loss_SSIM()
criterion_sam = Loss_SAM()
data_processing = Data_Process(
    spatial_keep_ratio=opt.spatial_keep_ratio,
    spatial_mask_seed=opt.spatial_mask_seed,
    spatial_mask_cache_root=opt.spatial_mask_cache_root,
)


def build_ratio_tag():
    return f"ratio_{opt.spatial_keep_ratio:.2f}"


def resolve_pretrained_model_path():
    if opt.pretrained_model_path is not None:
        return opt.pretrained_model_path
    return os.path.join(opt.checkpoint_root, build_ratio_tag(), opt.checkpoint_name)


def load_mask():
    mask_init = load_mat_compat(opt.mask_path)["mask"]
    print("mask_init:", mask_init.shape)
    if mask_init.ndim == 3 and mask_init.shape[0] > mask_init.shape[-1]:
        mask_init = np.transpose(mask_init, (2, 0, 1))
    mask = mask_init[
        :,
        opt.start_dir[0]:opt.start_dir[0] + opt.image_size[0],
        opt.start_dir[1]:opt.start_dir[1] + opt.image_size[1],
    ]
    mask = np.maximum(mask, 0)
    mask = mask / mask.max()
    mask = torch.from_numpy(mask).cuda()
    print("mask:", mask.dtype, mask.shape, mask.max(), mask.mean(), mask.min())
    return mask


def evaluate(test_loader, model, mask):
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    sam_meter = AverageMeter()

    test_pbar = tqdm(test_loader, desc="Test", leave=False)
    for hsis in test_pbar:
        hsis = hsis.cuda(non_blocking=True)
        batch_size = hsis.size(0)

        mask_patch = data_processing.get_fixed_center_mask_patches(
            mask=mask,
            image_size=opt.image_size,
            patch_size=opt.patch_size,
            batch_size=batch_size,
        )
        inputs, targets = data_processing.get_mos_hsi(
            hsi=hsis,
            mask=mask_patch,
            sigma=tuple(opt.sigma),
            mos_size=opt.patch_size[0],
            hsi_input_size=opt.patch_size[0],
            hsi_target_size=opt.patch_size[0],
        )

        with torch.no_grad():
            outputs = model(inputs, mask_patch)

            psnr = criterion_psnr(outputs, targets)
            ssim = criterion_ssim(outputs, targets)
            sam = criterion_sam(outputs, targets)

        psnr_meter.update(float(psnr), batch_size)
        ssim_meter.update(float(ssim), batch_size)
        sam_meter.update(float(sam), batch_size)
        test_pbar.set_postfix(
            psnr=f"{psnr_meter.avg:.4f}",
            ssim=f"{ssim_meter.avg:.4f}",
            sam=f"{sam_meter.avg:.4f}",
        )

    return psnr_meter.avg, ssim_meter.avg, sam_meter.avg


def main():
    cudnn.benchmark = True
    pretrained_model_path = resolve_pretrained_model_path()
    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_model_path}")

    print(f"checkpoint: {pretrained_model_path}")
    print(f"test_data_path: {opt.test_data_path}")

    mask = load_mask()
    model = model_generator(opt.method, pretrained_model_path)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")

    if torch.cuda.is_available():
        criterion_psnr.cuda()
        criterion_ssim.cuda()
        criterion_sam.cuda()

    test_data = HyperspectralDataset(root_dir=opt.test_data_path)
    print("len(test_data):", len(test_data))
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    avg_psnr, avg_ssim, avg_sam = evaluate(test_loader, model, mask)

    print("\nTest Results")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"SAM:  {avg_sam:.4f}")


if __name__ == "__main__":
    main()
