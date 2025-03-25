import os
import numpy as np
import random
import torch
from pathlib import Path
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F

import utils.utils_video as utils_video
import time
from PIL import Image

class VideoRecurrentTrainDataset(data.Dataset):
    """Video dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_XXX_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    720p_240fps_1 100 (720,1280,3)
    720p_240fps_3 100 (720,1280,3)
    ...

    Key examples: "720p_240fps_1/00000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_flow (str, optional): Data root path for flow.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(VideoRecurrentTrainDataset, self).__init__()
        self.opt = opt
        self.scale = opt.get('scale', 1)
        self.gt_size = opt.get('gt_size', 256)
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        self.filename_tmpl = opt.get('filename_tmpl', '04d') ##
        self.filename_ext = opt.get('filename_ext', 'npy') ##
        self.num_frame = opt['num_frame']

        keys = []
        total_num_frames = [] # some clips may not have 100 frames
        start_frames = [] # some clips may not start from 00000

        self.gt_path = os.path.join(self.gt_root, 'GT')
        self.lq_path = os.path.join(self.lq_root, 'Input')

        self.video_list_gt = [f.path for f in os.scandir(self.gt_path) if f.is_dir()]
        # self.video_list_lq = [f.path for f in os.scandir(self.lq_path) if f.is_dir()]
        numbers = [int(os.path.basename(path)) for path in self.video_list_gt]
        self.num_npy_files = len([filename for filename in os.listdir(self.video_list_gt[0]) if filename.endswith('.'+self.filename_ext)])
        print(">>>>>>>>>>>>>>>>>>>> NUM OF VIDEOS FOR TRAINING:", "/", len(self.video_list_gt))
        
        for folder in numbers:#self.video_list:
            keys.extend([f'{folder:{self.filename_tmpl}}'])
            total_num_frames.extend([self.num_npy_files])
            start_frames.extend([0])
        
        self.keys = []
        self.total_num_frames = [] # some clips may not have 100 frames
        self.start_frames = []
        self.test_mode = opt['test_mode']
        
        for i, v in zip(range(len(keys)), keys):
            self.keys.append(keys[i])
            self.total_num_frames.append(total_num_frames[i])
            self.start_frames.append(start_frames[i])
        
        self.io_backend_opt = opt['io_backend']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        print(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        # begin = time.time()

        key = self.keys[index]
        total_num_frames = self.total_num_frames[index]
        start_frames = self.start_frames[index]
        
        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = random.randint(0, total_num_frames - self.num_frame)
        
        endmost_start_frame_idx = start_frames + total_num_frames - self.num_frame * interval
        if start_frame_idx > endmost_start_frame_idx:
            start_frame_idx = random.randint(start_frames, endmost_start_frame_idx)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            img_lq_path = key +  f'/{neighbor:{self.filename_tmpl}}' + f'.{self.filename_ext}'
            img_gt_path = key + f'/{neighbor:{self.filename_tmpl}}' + f'.{self.filename_ext}'

            if self.filename_ext == 'npy':
                # print(img_lq_path, img_gt_path)
                img_lq = np.load(os.path.join(self.lq_root, 'Input', img_lq_path)) / 255.0
                img_lqs.append(img_lq)

                img_gt = np.load(os.path.join(self.gt_root, 'GT', img_gt_path)) / 255.0
                img_gts.append(img_gt)

            elif self.filename_ext == 'png':
                img_lq = Image.open(os.path.join(self.lq_root, 'Input', img_lq_path))
                img_lq = np.array(img_lq) / 255.0
                img_lqs.append(img_lq)

                img_gt = Image.open(os.path.join(self.lq_root, 'GT', img_lq_path))
                img_gt = np.array(img_gt) / 255.0
                img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = utils_video.paired_random_crop(img_gts, img_lqs, self.gt_size, self.scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = utils_video.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])
        img_results = utils_video.img2tensor(img_results,bgr2rgb=False,float32=True)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)
        img_gts_L1 = F.interpolate(img_gts, scale_factor=0.5, mode='bilinear', align_corners=False)
        img_gts_L2 = F.interpolate(img_gts_L1, scale_factor=0.5, mode='bilinear', align_corners=False)
        img_gts_L3 = F.interpolate(img_gts_L2, scale_factor=0.5, mode='bilinear', align_corners=False)

        # end = time.time()
        # print("!!!!!!! Elapsed time for DataLoader:", end-begin)
        # print(f"{torch.distributed.get_rank()=} {t6 - t5=:.6f} {t5 - t4=:.6f} {t4 - t3=:.6f} {t3 - t2=:.6f} {t2 - t1=:.6f}")

        # return {'L': img_lqs, 'H': img_gts, 'key': key}
        return {'L': img_lqs, 'H': img_gts,  'H1': img_gts_L1,  'H2': img_gts_L2,  'H3': img_gts_L3, 'key': key}


    def __len__(self):
        return len(self.keys)

