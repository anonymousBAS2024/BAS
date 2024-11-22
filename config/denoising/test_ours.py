import argparse
import logging
import os.path
import shutil
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
from IPython import embed
import lpips

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default="config/denoising/options/test/ours.yml", help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result_denoising")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result_denoising")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device
lpips_fn = lpips.LPIPS(net='alex').to(device)

sde = util.Gaussian_Diffusion(T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)

degrad_sigma = opt["degradation"]["sigma"]

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["lpips"] = []
    test_results["fid"] = []
    test_times = []

    for i, test_data in enumerate(test_loader):
        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        GT = test_data["GT"]
        LQ = util.add_noise(GT, degrad_sigma)

        noisy_state = sde.noise_state(LQ)

        model.feed_data(noisy_state, LQ, GT)
        tic = time.time()
        model.test_visual_x0(sde = sde, save_states=True)
 

        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        SR_img = visuals["Output"]
        output = util.tensor2img(SR_img.squeeze())  # uint8
        LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
        GT_ = util.tensor2img(visuals["GT"].squeeze())  # uint8
        
        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, img_name + ".png")
        util.save_img(output, save_img_path)

        # remove it if you only want to save output images
        LQ_img_path = os.path.join(dataset_dir, img_name + "_LQ.png")
        GT_img_path = os.path.join(dataset_dir, img_name + "_HQ.png")
        util.save_img(LQ_, LQ_img_path)
        util.save_img(GT_, GT_img_path)

        if need_GT:  
            gt_img = GT_ / 255.0
            sr_img = output / 255.0
            cropped_sr_img = sr_img
            cropped_gt_img = gt_img


            psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
            
            # Calculate FID
            

            # Create directories for the SR and GT images
            sr_dir = 'temp_sr'
            gt_dir = 'temp_gt'
            os.makedirs(sr_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)
            sr_image_path = os.path.join(sr_dir, 'sr_image.png')
            gt_image_path = os.path.join(gt_dir, 'gt_image.png')
            util.save_img(cropped_sr_img * 255, sr_image_path)  # Save SR image
            util.save_img(cropped_gt_img * 255, gt_image_path)  # Save GT image

            fid = util.calculate_fid(gt_dir, sr_dir)
            
            lp_score = lpips_fn(
                GT.to(device) * 2 - 1, SR_img.to(device) * 2 - 1).squeeze().item()

            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)
            test_results["lpips"].append(lp_score)
            test_results["fid"].append(fid)

            if len(gt_img.shape) == 3:
                    logger.info(
                        "img{:3d}:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; LPIPS: {:.6f}; FID: {:.6f}.".format(
                            i, img_name, psnr, ssim, lp_score, fid
                        )
                    )
            else:
                logger.info(
                    "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}.".format(
                        img_name, psnr, ssim
                    )
                )

        else:
            logger.info(img_name)


    ave_lpips = sum(test_results["lpips"]) / len(test_results["lpips"])
    ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
    ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
    ave_fid = sum(test_results["fid"]) / len(test_results["fid"])
    logger.info(
        "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
            test_set_name, ave_psnr, ave_ssim
        )
    )

    logger.info(
            "----average LPIPS\t: {:.6f}\n".format(ave_lpips)
        )
    
    logger.info(
            "----average FID\t: {:.6f}\n".format(ave_fid)
        )

    print(f"average test time: {np.mean(test_times):.4f}")


    shutil.rmtree(sr_dir)
    shutil.rmtree(gt_dir)
