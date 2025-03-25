import cv2
import mmcv
import torch
import numpy as np
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop, totensor
from basicsr.data.util import (paired_paths_from_meta_info_file,
                               paired_paths_PSF_from_meta_info_file,
                               paired_paths_from_folder,
                               paired_paths_from_lmdb)
from basicsr.utils import FileClient
from basicsr.utils.registry import DATASET_REGISTRY
import random
import time
from pathlib import Path
import torchvision
from natsort import natsorted

#
# UDCUNet's PairedImgPSFNpyDataset is actually PairedImgNpyDataset in DISCNet
# I guess they didn't read basicsr documentation.
#


@DATASET_REGISTRY.register()
class PairedImgPSFNpyTestDataset(data.Dataset):
    """Paired image dataset with its corresponding PSF.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc)
    and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal and vertical flips.
            use_rot (bool): Use rotation (use transposing h and w for
                implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

   
        for folder_name, folder_opt in opt['folders'].items():
            self.gt_folder, self.lq_folder =Path(folder_opt['dataroot_gt']), Path(folder_opt['dataroot_lq'])
            self.meta_info_file = folder_opt['meta_info_file'] 
      

        self.frame_num = opt['frame_num']
     
        self.keys = []
        with open(self.meta_info_file, 'r') as fin:
            for line in fin:
                folder, _ = line.split(' ')
                self.keys.extend(
                    [f'{folder}/{i:04d}' for i in range(int(self.frame_num))])
        
        self.keys = natsorted(self.keys)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        


    def _tonemap(self, x, type='simple', range=1023.):
        if type == 'mu_law':
            norm_x = x / x.max()
            mapped_x = np.log(1 + 10000 * norm_x) / np.log(1 + 10000)
        elif type == 'simple':
            mapped_x = x / (x + 0.25)
        elif type == 'same':
            mapped_x = x
        elif type == "norm":
            mapped_x = x / range
        else:
            raise NotImplementedError(
                'tone mapping type [{:s}] is not recognized.'.format(type))
        return mapped_x

    def _expand_dim(self, x):
        # expand dimemsion if images are gray.
        if x.ndim == 2:
            return x[:, :, None]
        else:
            return x

    # I suppose 4-channeled SIT will still work with this dataloader.
    # Other parts in the code will be examined further...
    def __getitem__(self, index):
        # begin = time.time()
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        lq_map_type = self.opt['lq_map_type']
        gt_map_type = self.opt['gt_map_type']
        lq_map_range = self.opt.get("lq_map_range")
        gt_map_range = self.opt.get("gt_map_range")
        order = self.opt.get("order")        
        
        key = self.keys[index]
        
        
        clip_name, frame_name = key.split('/')  # key example: 000/00000000
 
    
        img_lq_path = self.lq_folder / clip_name / f'{frame_name}.npy' 
        img_gt_path = self.gt_folder / clip_name / f'{frame_name}.npy' 
                
      
        img_lq = np.load(img_lq_path)
        img_gt = np.load(img_gt_path)

        if order == "cwh":
            img_lq = img_lq.transpose(2, 1, 0)
            img_gt = img_gt.transpose(2, 1, 0)
        
        img_lq = self._tonemap(img_lq, type=lq_map_type, range=lq_map_range)
        img_gt = self._tonemap(img_gt, type=gt_map_type, range=gt_map_range)
  
        crop_scale = self.opt.get('crop_scale', None)
        
        # augmentation
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                img_gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = totensor(
            [img_gt, img_lq], bgr2rgb=False, float32=True)
        
        if self.opt['phase'] != 'train':
            centercrop = torchvision.transforms.CenterCrop((1056,1896))
            img_gt = centercrop(img_gt)
            img_lq = centercrop(img_lq)
        
      
       
        
        # print(self.opt['phase'],"lq gt shape",img_lq.shape,img_gt.shape)
        
        
        # for full image patch-wise inference
        # 0_top_left
        # img_gt = img_gt[ :, :2048, :1536] # 2048 1536
        # img_lq = img_lq[ :, :2048, :1536]
        
        # # 1_top_right
        # img_gt = img_gt[ :, :2048, 1024:] # 2048 1536
        # img_lq = img_lq[ :, :2048, 1024:]
        
        # # 2_bot_left
        # img_gt = img_gt[ :, 1536:, :1536] # 2048 1536
        # img_lq = img_lq[ :, 1536:, :1536]
        
        # # 3_bot_right
        # img_gt = img_gt[ :, 1536:, 1024:] # 2048 1536
        # img_lq = img_lq[ :, 1536:, 1024:]
        
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': str(img_lq_path),
            'gt_path': str(img_gt_path),
        }

    def __len__(self):
        # return len(self.paths)
        return len(self.keys)
