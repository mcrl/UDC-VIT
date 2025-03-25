import os
import glob
import numpy as np
import torch
from os import path as osp
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import utils_ddrnet.utils_video as utils_video

from pathlib import Path
import random

class VideoRecurrentValDataset4Vit(data.Dataset):

    def __init__(self, opt):
        super(VideoRecurrentValDataset4Vit, self).__init__()
        self.opt = opt
        self.scale = opt.get('scale', 1)
        self.gt_size = opt.get('gt_size', 256)
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.filename_tmpl = opt.get('filename_tmpl', '04d')
        self.filename_ext = opt.get('filename_ext', 'npy')
        self.num_frame = opt['num_frame']

        keys = []
        total_num_frames = [] # some clips may not have 100 frames
        start_frames = [] # some clips may not start from 00000

        self.video_list_gt = [f.path for f in os.scandir(self.gt_root) if f.is_dir()]
        numbers = [int(os.path.basename(path)) for path in self.video_list_gt]
        self.num_npy_files = len([filename for filename in os.listdir(self.video_list_gt[0]) if filename.endswith('.npy')])
        print(">>>>>>>>>>>>>>>>>>>> NUM OF VIDEOS FOR VALIDATION:", "/", len(self.video_list_gt))

        for folder in numbers:
            keys.extend([f'{folder:{self.filename_tmpl}}'])
            total_num_frames.extend([self.num_npy_files])
            start_frames.extend([0])

        self.keys = []
        self.total_num_frames = []
        self.start_frames = []
        
        for i, v in zip(range(len(keys)), keys):
            for j in range(total_num_frames[i] - self.num_frame):
                self.keys.append(keys[i])
                self.total_num_frames.append(self.num_frame)
                self.start_frames.append(j)
                
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        print(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        total_num_frames = self.total_num_frames[index]
        start_frames = self.start_frames[index]
        
        # determine the neighboring frames
        interval = random.choice(self.interval_list)
        start_frame_idx = start_frames
        end_frame_idx = start_frames + total_num_frames
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # get the neighboring LQ and GT frames

        # img_lqs = []
        # for neighbor in neighbor_list:
        #     img_lq_path = key +  f'/{neighbor:{self.filename_tmpl}}' + f'.{self.filename_ext}'
        #     img_lq = self.file_client.get(os.path.join(self.lq_root, img_lq_path), 'lq') / 255.0
        #     img_lqs.append(img_lq)
        # img_lqs = utils_video.img2tensor(img_lqs,bgr2rgb=False,float32=True)
        # simg_lqs = torch.stack(img_lqs, dim=0)


        # img_gts = []
        # img_gt_path = key + f'/{neighbor_list[2]:{self.filename_tmpl}}' + f'.{self.filename_ext}'
        # img_gt = self.file_client.get(os.path.join(self.gt_root, img_gt_path), 'gt') / 255.0
        # img_gts.append(img_gt)
        # img_gts = utils_video.img2tensor(img_gts,bgr2rgb=False,float32=True)
        # simg_gts = torch.stack(img_gts, dim=0)
        # rimg_gts = simg_gts[2]

        # img_lqs = []
        # img_gts = []
        # for neighbor in neighbor_list:
        #      # get LQ
        #     img_lq_path = key +  f'/{neighbor:{self.filename_tmpl}}' + f'.{self.filename_ext}'
        #     img_lq = self.file_client.get(os.path.join(self.lq_root, img_lq_path), 'lq') / 255.0
        #     img_lqs.append(img_lq)

        #     # get GT
        #     img_gt_path = key + f'/{neighbor:{self.filename_tmpl}}' + f'.{self.filename_ext}'
        #     img_gt = self.file_client.get(os.path.join(self.gt_root, img_gt_path), 'gt') / 255.0
        #     img_gts.append(img_gt)

        # # randomly crop
        # img_gts, img_lqs = utils_video.paired_random_crop(img_gts, img_lqs, self.gt_size, self.scale, img_gt_path)
        
        # img_lqs = utils_video.img2tensor(img_lqs,bgr2rgb=False,float32=True)
        # simg_lqs = torch.stack(img_lqs, dim=0)

        # img_gts = utils_video.img2tensor(img_gts,bgr2rgb=False,float32=True)
        # simg_gts = torch.stack(img_gts, dim=0)
        # rimg_gts = simg_gts[2]

        # return {'L': simg_lqs, 'H':rimg_gts}

        # get LQ
        img_lqs = []
        for neighbor in neighbor_list:
            img_lq_path = key +  f'/{neighbor:{self.filename_tmpl}}' + f'.{self.filename_ext}'
            img_lq = self.file_client.get(os.path.join(self.lq_root, img_lq_path), 'lq') / 255.0
            # img_lq = np.load(os.path.join(self.lq_root, img_lq_path)) / 255.0
            img_lqs.append(img_lq)

        # get GT
        img_gts = []
        img_gt_path = key + f'/{neighbor_list[2]:{self.filename_tmpl}}' + f'.{self.filename_ext}'
        img_gt = self.file_client.get(os.path.join(self.gt_root, img_gt_path), 'gt') / 255.0
        # img_gt = np.load(os.path.join(self.gt_root, img_gt_path)) / 255.0
        img_gts.append(img_gt)
        
        # randomly crop
        img_gts, img_lqs = utils_video.paired_center_crop(img_gts, img_lqs, self.gt_size, self.scale, img_gt_path)

        img_lqs.extend(img_gts)
        img_results = utils_video.augment(img_lqs, hflip=False, vflip=False, rotation=False, flows=None)
        img_results = utils_video.img2tensor(img_results, bgr2rgb=False, float32=True)

        simg_lqs = torch.stack(img_results[:len(img_lqs)-1], dim=0)

        img_lqs = simg_lqs.reshape(simg_lqs.size(0) * simg_lqs.size(1), simg_lqs.size(2), simg_lqs.size(3))
        
        img_gts = img_results[-1]

        return {'L': img_lqs, 'H':img_gts}
        
    def __len__(self):
        return len(self.keys)
