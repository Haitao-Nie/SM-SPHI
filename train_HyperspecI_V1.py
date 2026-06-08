import torch
import argparse
import os
import time
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from getdataset import HyperspectralDataset
from my_utils import AverageMeter, initialize_logger, save_checkpoint, Loss_RMSE, Loss_PSNR, Loss_TV, Loss_MRAE, Loss_SAM, load_mat_compat
from DataProcess import Data_Process
import torch.utils.data
from architecture import model_generator
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Model training of HyperspecI-V1")
parser.add_argument("--method", type=str, default='V1_srnet_pconv', help='Model')
parser.add_argument('--train_batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--val_batch_size', type=int, default=64, help='validation batch size')
parser.add_argument("--end_epoch", type=int, default=200, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--gpu_id", type=str, default='0', help='select gpu')
parser.add_argument("--pretrained_model_path", type=str, default=None, help='pre-trained model path')
parser.add_argument("--sigma", type=float, default=(0, 1 / 255, 2/255, 3/255), help="Sigma of Gaussian Noise")
parser.add_argument("--mask_path", type=str, default='./MASK/Mask_HyperspecI_V1.mat', help='path of calibrated sensing matrix')
parser.add_argument("--output_folder", type=str, default='./exp/HyperspecI_V1/', help='output path')
parser.add_argument("--start_dir", type=int, nargs=2, default=(0, 0), help="size of test image coordinate")
parser.add_argument("--image_size", type=int, nargs=2, default=(2048, 2048), help="size of test image")
parser.add_argument("--patch_size", type=int, nargs=2, default=(64, 64), help="HSI patch size")
parser.add_argument("--spatial_keep_ratio", type=float, default=0.3, help="fraction of spatial locations kept in the binary spatial gate")
parser.add_argument("--spatial_mask_seed", type=int, default=42, help="seed for deterministic spatial gate generation")
parser.add_argument("--spatial_mask_cache_root", type=str, default="./MASK/blue_noise_masks", help="folder used to load/save deterministic blue-noise-like spatial masks")
parser.add_argument("--train_data_path", type=str, default="./ICVL_64/train/", help='path datasets')
parser.add_argument("--valid_data_path", type=str, default="./ICVL_64/val/", help='path datasets')



opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_mrae = Loss_MRAE()
criterion_sam = Loss_SAM()
criterion_tv = Loss_TV(TVLoss_weight=float(0.5))
data_processing = Data_Process(
    spatial_keep_ratio=opt.spatial_keep_ratio,
    spatial_mask_seed=opt.spatial_mask_seed,
    spatial_mask_cache_root=opt.spatial_mask_cache_root,
)

def load_mask():
    mask_init = load_mat_compat(opt.mask_path)['mask']
    print('mask_init:', mask_init.shape)
    # Ensure mask is channel-first: [C, H, W]
    if mask_init.ndim == 3 and mask_init.shape[0] > mask_init.shape[-1]:
        mask_init = np.transpose(mask_init, (2, 0, 1))
    mask = mask_init[
        :,
        opt.start_dir[0]:opt.start_dir[0] + opt.image_size[0],
        opt.start_dir[1]:opt.start_dir[1] + opt.image_size[1]
    ]
    mask = np.maximum(mask, 0)
    mask = mask / mask.max()
    mask = torch.from_numpy(mask).cuda()
    print('mask:', mask.dtype, mask.shape, mask.max(), mask.mean(), mask.min())
    return mask

def main():
    cudnn.benchmark = True
    mask = load_mask()

    print("\nloading dataset ...")
    train_data = HyperspectralDataset(root_dir=opt.train_data_path)
    print('len(train_data):', len(train_data))
    val_data = HyperspectralDataset(root_dir=opt.valid_data_path)
    print('len(valid_data):', len(val_data))
    output_path = opt.output_folder

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = model_generator(opt.method, opt.pretrained_model_path)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    if torch.cuda.is_available():
        criterion_rmse.cuda()
        criterion_psnr.cuda()
        criterion_tv.cuda()
        criterion_mrae.cuda()
        
    start_epoch = 0

    train_loader = DataLoader(dataset=train_data, batch_size=opt.train_batch_size, shuffle=True, num_workers=2,
                pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(dataset=val_data, batch_size=opt.val_batch_size, shuffle=False, num_workers=2,
                pin_memory=True, persistent_workers=True)
    per_epoch_iteration = len(train_loader)
    total_iteration = per_epoch_iteration * opt.end_epoch
    iteration = start_epoch * per_epoch_iteration

    #opt.init_lr
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.init_lr,
                                 betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration - iteration, eta_min=1e-6)

    log_dir = os.path.join(output_path, 'train.log')
    logger = initialize_logger(log_dir)

    record_rmse_loss = 10000
    strat_time = time.time()
    
    for epoch in range(start_epoch, opt.end_epoch):
        model.train()
        losses = AverageMeter()

        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{opt.end_epoch}", leave=False)
        for i, (HSIs) in enumerate(train_pbar):

            HSIs = HSIs.cuda()
            mask_patch = data_processing.get_fixed_center_mask_patches(mask=mask, image_size=opt.image_size, patch_size=opt.patch_size, batch_size=opt.train_batch_size)
            #Generate the measurements using traning HSIs and selected sub-pattern
            inputs, targets = data_processing.get_mos_hsi(hsi=HSIs, mask=mask_patch, sigma=opt.sigma, mos_size=opt.patch_size[0], hsi_input_size=opt.patch_size[0], hsi_target_size=opt.patch_size[0])

            inputs = Variable(inputs)
            targets = Variable(targets)
 
            lr = optimizer.param_groups[0]['lr']
            outputs = model(inputs, mask_patch)

            #calculate the hybrid loss
            loss_rmse = criterion_rmse(outputs, targets)
            loss_tv = criterion_tv(outputs, targets) 
            loss_mrae = criterion_mrae(outputs, targets) * 0.2
            loss = loss_rmse + loss_tv + loss_mrae
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad() 
            scheduler.step() 
                
            losses.update(loss.data)
            iteration = iteration + 1
            train_pbar.set_postfix({
                "lr": f"{lr:.2e}",
                "loss": f"{float(losses.avg):.6f}"
            })

        end_time = time.time()
        epoch_time = end_time - strat_time
        strat_time = time.time()
        rmse_loss, psnr_loss, mrae_loss, sam_loss = Validate(val_loader, model, mask)

        # Save model
        if torch.abs(record_rmse_loss - rmse_loss) < 0.0001 or rmse_loss < record_rmse_loss or iteration % 10000 == 0:
            print(f'Saving to {output_path}')
            save_checkpoint(output_path, (epoch + 1), iteration, model, optimizer)
            if rmse_loss < record_rmse_loss:
                record_rmse_loss = rmse_loss
        # print loss
        print(" Epoch[%06d], Time[%06d], learning rate: %.9f, Train Loss: %.9f, "
              "Val RMSE: %.9f, Val PSNR: %.9f, Val MRAE: %.9f, Val SAM: %.9f "
              % (epoch + 1, epoch_time, lr, losses.avg, rmse_loss, psnr_loss, mrae_loss, sam_loss))

        logger.info(" Epoch[%06d], Time[%06d], learning rate: %.9f, Train Loss: %.9f, "
              "Val RMSE: %.9f, Val PSNR: %.9f, Val MRAE: %.9f, Val SAM: %.9f "
              % (epoch + 1, epoch_time, lr, losses.avg, rmse_loss, psnr_loss, mrae_loss, sam_loss))
        
def Validate(val_loader, model, mask):
    model.eval()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_mrae = AverageMeter()
    losses_sam = AverageMeter()
    val_pbar = tqdm(val_loader, desc="Validate", leave=False)
    for i, (HSIs) in enumerate(val_pbar):
        HSIs = HSIs.cuda()
        batch_size = HSIs.size(0)

        mask_patch = data_processing.get_fixed_center_mask_patches(mask=mask, image_size=opt.image_size, patch_size=opt.patch_size, batch_size=batch_size)
        
        #Generate the measurements using traning HSIs and selected sub-pattern
        inputs, targets = data_processing.get_mos_hsi(hsi=HSIs, mask=mask_patch, sigma=opt.sigma, mos_size=opt.patch_size[0], hsi_input_size=opt.patch_size[0], hsi_target_size=opt.patch_size[0])

        with torch.no_grad():
            outputs = model(inputs, mask_patch)

            loss_rmse = criterion_rmse(outputs, targets)
            loss_psnr = criterion_psnr(outputs, targets)
            loss_mrae = criterion_mrae(outputs, targets)
            loss_sam = criterion_sam(outputs, targets)
            losses_psnr.update(loss_psnr.data)
            losses_rmse.update(loss_rmse.data)
            losses_mrae.update(loss_mrae.data)
            losses_sam.update(loss_sam.data)
            val_pbar.set_postfix({
                "rmse": f"{float(losses_rmse.avg):.6f}",
                "psnr": f"{float(losses_psnr.avg):.4f}"
            })

    return losses_rmse.avg, losses_psnr.avg, losses_mrae.avg, losses_sam.avg


if __name__ == '__main__':
    main()


