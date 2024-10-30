# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from tensorfn import load_config as DiffConfig
import numpy as np
from config.diffconfig import DiffusionConfig, get_model_conf
import torch.distributed as dist
import os, glob, cv2, time, shutil
from models.unet_autoenc import BeatGANsAutoencConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
import torchvision.transforms as transforms
import torchvision
import sys

np.set_printoptions(threshold=sys.maxsize)

class Predictor():
    def __init__(self):
        """Load the model into memory to make running multiple predictions efficient"""

        conf = DiffConfig(DiffusionConfig, './config/diffusion.conf', show=False)

        self.model = get_model_conf().make_model()

        if torch.cuda.is_available():
            print("cuda를 사용합니다.")
            device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     device = torch.device('mps')
        #     print('mps 사용합니다.')
        else:
            device = torch.device('cpu')
            print('cpu 사용합니다.')

        ckpt = torch.load("checkpoints/last.pt", map_location=device)
        self.model.load_state_dict(ckpt["ema"])
        self.model = self.model.to(device)
        self.model.eval()

        self.betas = conf.diffusion.beta_schedule.make()
        self.diffusion = create_gaussian_diffusion(self.betas, predict_xstart = False)#.to(device)
        
        self.pose_list = np.load('data/deepfashion_256x256/target_pose/joint_data.npy')
        
        # impl here
        # np cast to float32
        # norm image size

        self.pose_list = self.pose_list.astype(np.float32)

        self.transforms = transforms.Compose([transforms.Resize((256,256), interpolation=Image.BICUBIC),
                            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])
    def predict_pose(
        self,
        image,
        num_poses=1,
        sample_algorithm='ddim',
        nsteps=100,

        ):
        """Run a single prediction on the model"""

        if torch.cuda.is_available():
            print("cuda를 사용합니다.")
            device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     device = torch.device('mps')
        #     print('mps 사용합니다.')
        else:
            device = torch.device('cpu')
            print('cpu 사용합니다.')

        src = Image.open(image)
        src = self.transforms(src).unsqueeze(0).to(device)
        print("----------")

        print(src.dtype)
        print(src.shape)
        print(self.pose_list.shape)
        print(self.pose_list.dtype)

        print(src.dtype)

        tgt_pose = torch.stack([transforms.ToTensor()(self.pose_list).to(device)], 0)
        print("num_poses:", num_poses)
        src = src.repeat(num_poses,1, 1, 1)
        print(src.shape)

        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, tgt_pose], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).to(device)
            print(noise.shape)
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.to(device), [src, tgt_pose])
            samples = xs[-1].to(device)


        samples_grid = torch.cat([src[0],torch.cat([samps for samps in samples], -1)], -1)
        samples_grid = (torch.clamp(samples_grid, -1., 1.) + 1.0)/2.0
        pose_grid = torch.cat([torch.zeros_like(src[0]),torch.cat([samps[:3] for samps in tgt_pose], -1)], -1)

        output = torch.cat([1-pose_grid, samples_grid], -2)

        numpy_imgs = output.unsqueeze(0).permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        # Image.fromarray(fake_imgs[0]).save('output.png')
        fake_imgs[0].save('output.png')


    def predict_appearance(
        self,
        image,
        ref_img,
        ref_mask,
        ref_pose,
        sample_algorithm='ddim',
        nsteps=100,

        ):
        """Run a single prediction on the model"""

        src = Image.open(image)
        src = self.transforms(src).unsqueeze(0).cuda()
        
        ref = Image.open(ref_img)
        ref = self.transforms(ref).unsqueeze(0).cuda()

        mask = transforms.ToTensor()(Image.open(ref_mask)).unsqueeze(0).cuda()
        pose =  transforms.ToTensor()(np.load(ref_pose)).unsqueeze(0).cuda()


        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, pose, ref, mask], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, pose, ref, mask], diffusion=self.diffusion)
            samples = xs[-1].cuda()


        samples = torch.clamp(samples, -1., 1.)

        output = (torch.cat([src, ref, mask*2-1, samples], -1) + 1.0)/2.0

        numpy_imgs = output.permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)   
        Image.fromarray(fake_imgs[0]).save('output.png')

if __name__ == "__main__":


    obj = Predictor()

    obj.predict_pose(image='test.jpg', num_poses=4, sample_algorithm = 'ddim',  nsteps = 50)
    
    # ref_img = "data/deepfashion_256x256/target_edits/reference_img_0.png"
    # ref_mask = "data/deepfashion_256x256/target_mask/lower/reference_mask_0.png"
    # ref_pose = "data/deepfashion_256x256/target_pose/reference_pose_0.npy"

    # #obj.predict_appearance(image='test.jpg', ref_img = ref_img, ref_mask = ref_mask, ref_pose = ref_pose, sample_algorithm = 'ddim',  nsteps = 50)
