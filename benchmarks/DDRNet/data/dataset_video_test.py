import glob
import torch
import os
from os import path as osp
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import numpy as np

import torch.nn.functional as F
import utils.utils_video as utils_video
import time
from PIL import Image

class VideoRecurrentTestDataset(data.Dataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames. Modified from
    https://github.com/xinntao/BasicSR/blob/master/basicsr/data/reds_dataset.py

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoRecurrentTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data'] 
        # self.gt_size = opt.get('gt_size', [1080,1920]) 
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}

        self.imgs_lq, self.imgs_gt = {}, {}

        self.gt_path = os.path.join(self.gt_root, 'GT')
        self.lq_path = os.path.join(self.lq_root, 'Input')

        self.video_list_gt = [f.path for f in os.scandir(self.gt_path) if f.is_dir()]
        # self.video_list_lq = [f.path for f in os.scandir(self.lq_path) if f.is_dir()]
        numbers = [int(os.path.basename(path)) for path in self.video_list_gt]
        self.num_npy_files = len([filename for filename in os.listdir(self.video_list_gt[0]) if filename.endswith('.npy')])
        print(">>>>>>>>>>>>>>>>>>>> NUM OF VIDEOS FOR VALIDATION (OR TEST):", len(self.video_list_gt))

        for subfolder in numbers: #self.video_list:
            img_paths_lq=[]
            img_paths_gt=[]
            for j in range(self.num_npy_files):
                subfolder = str(subfolder).zfill(4)
                key = subfolder+'/'+str(j).zfill(4)+'.npy'
                img_paths_lq.append(key)
                img_paths_gt.append(key)
            max_idx = len(img_paths_lq)
            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['gt_path'].extend(img_paths_gt)
            self.data_info['folder'].extend([subfolder] * max_idx)
            # for i in range(max_idx):
            #     self.data_info['idx'].append(f'{i}/{max_idx}')
            # border_l = [0] * max_idx
            # for i in range(self.opt['num_frame'] // 2):
            #     border_l[i] = 1
            #     border_l[max_idx - i - 1] = 1
            # self.data_info['border'].extend(border_l)
            self.imgs_lq[subfolder] = img_paths_lq
            self.imgs_gt[subfolder] = img_paths_gt

        # Find unique folder strings
        self.io_backend_opt = opt['io_backend']
        self.folders = sorted(list(set(self.data_info['folder'])))
    
    def __getitem__(self, index):

        
        folder = self.folders[index]
        img_lqs = []
        img_gts = []
        for ind in range(self.num_npy_files):
            img_lq_path = folder + '/' + str(ind).zfill(4) + '.npy' #f'.{self.filename_ext}'
            img_gt_path = folder + '/' + str(ind).zfill(4) + '.npy' #f'.{self.filename_ext}'

            # if self.filename_ext == 'npy':
            img_lq = np.load(os.path.join(self.lq_root, 'Input',img_lq_path)) / 255.0
            img_gt = np.load(os.path.join(self.gt_root, 'GT', img_gt_path)) / 255.0
            img_lqs.append(img_lq)
            img_gts.append(img_gt)
            
            # elif self.filename_ext == 'png':
            #     img_lq = Image.open(os.path.join(self.lq_root, 'Input', img_lq_path))
            #     img_lq = np.array(img_lq) / 255.0
            #     img_lqs.append(img_lq)

            #     img_gt = Image.open(os.path.join(self.lq_root, 'GT', img_lq_path))
            #     img_gt = np.array(img_gt) / 255.0
            #     img_gts.append(img_gt)

        
        
        img_lqs = utils_video.img2tensor(img_lqs,bgr2rgb=False,float32=True)
        img_gts = utils_video.img2tensor(img_gts,bgr2rgb=False,float32=True)
        img_lqs = torch.stack(img_lqs, dim=0)
        img_gts = torch.stack(img_gts, dim=0)
        img_gts_L1 = F.interpolate(img_gts, scale_factor=0.5, mode='bilinear', align_corners=False)
        img_gts_L2 = F.interpolate(img_gts_L1, scale_factor=0.5, mode='bilinear', align_corners=False)
        img_gts_L3 = F.interpolate(img_gts_L2, scale_factor=0.5, mode='bilinear', align_corners=False)
        return {
            'L': img_lqs,
            'H': img_gts,
            'H1': img_gts_L1,
            'H2': img_gts_L2,
            'H3': img_gts_L3,
            'folder': folder,
            'lq_path': self.imgs_lq[folder],
        }

    def __len__(self):
        return len(self.folders)



    # def __getitem__(self, index):
    #     folder = self.folders[index]

    #     if self.cache_data:
    #         imgs_lq = self.imgs_lq[folder]
    #     else:
    #         imgs_lq = utils_video.read_img_seq(self.imgs_lq[folder])

    #     return {
    #         'L': imgs_lq,
    #         'folder': folder,
    #         'lq_path': self.imgs_lq[folder],
    #     }
