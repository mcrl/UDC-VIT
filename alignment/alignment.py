import os
import sys
import natsort
import glob
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import torch
import rawpy as rp
from multiprocessing import Pool, Queue, Manager
import time


'''
Pre-processing for taken images of input and groundtruth.
This function first crop the groundtruth images with the size (target_h, target_w).
Then it shifts input images so that the loss between input and groundtruth becomes minimum.
User should set the parameters below:
  - target_h: the height of post-processed image
  - target_w: the width of post-processed image
  - max_shift: how much pixels will be shifted and compare the loss between img_input_crop and img_gt_crop
After the alignment using this script, the alignment discrepancies are performed through crowdsourcing.
Finally, we measure the Percentage of Correct Keypoints (PCK)
'''

def load_png(dataset_list):
    dataset_list = natsort.natsorted(dataset_list)
    img_list = [None] * len(dataset_list)
    fname_list = [None] * len(dataset_list)
    for i in range(len(dataset_list)):
        fname_list[i] = dataset_list[i].split("/")[-1]
        img = Image.open(dataset_list[i]).convert("RGB")
        img_array = np.array(img)
        # img_array = img_array.reshape(2, 0, 1)
        # print(">>>>>>>>", img_array.shape)
        img_list[i] = img_array

    return img_list, fname_list


def save_png(img_data, file_path):
    img_data = img_data.astype(np.uint8)  # Ensure data type is suitable for saving as PNG
    img = Image.fromarray(img_data.transpose(1, 2, 0))  # Transpose to (H, W, C) for saving as image
    img.save(file_path)

    
def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def FFT_abs_L1(_input, gt):
    criterion = torch.nn.L1Loss()
    input_fft = abs( torch.fft.fft2(_input) )
    gt_fft = abs( torch.fft.fft2(gt) )
    loss_fft = criterion(input_fft, gt_fft)
    return loss_fft


def FFT_angle_L1(_input, gt):
    criterion = torch.nn.L1Loss()
    input_fft = torch.angle( torch.fft.fft2(_input))  
    gt_fft = torch.angle( torch.fft.fft2(gt) ) 
    loss_fft = criterion(input_fft, gt_fft)
    return loss_fft


def calc_losses(_lambda, img_gt, img_input):
    mse = float(torch.nn.functional.mse_loss(img_gt, img_input)) if _lambda[0] != 0 else 0
    fft_abs = FFT_abs_L1(img_input, img_gt) if _lambda[1] != 0 else 0
    fft_angle = FFT_angle_L1(img_input, img_gt) if _lambda[2] != 0 else 0
    return mse, fft_abs, fft_angle


def warning_shift_value(st_h, st_w, max_shift):
    if(st_h < max_shift or st_w < max_shift):
        print("This script is to crop and align from larger than 1792 x 1280 x 4 (e.g., 2016 x 1512 x 4).")
        print("Check your image size, target_h, target_w, and max_shift.")
        exit(0)


def write_log(log_file_path, log_message):
    retry_count, max_retry = 0, 10
    while retry_count < max_retry:
        try:
            with open(log_file_path, 'a') as f:
                f.write(log_message)
            break
        except Exception as e:
            print(f"Failed to open log file. Retry count: {retry_count + 1}")
            retry_count += 1
            time.sleep(0.1)

    if retry_count == max_retry:
        print("Failed to open log file after multiple attempts.")    

def pre_processing(data):
    list_input, list_gt, fname_input, fname_gt, output_dir, target_h, target_w, max_shift, _lambda, img_intensity, log_file_path, device_num = data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11]
    if img_intensity == 255:
        scale = 1.0
    elif img_intensity == 1:
        scale = 255.0
    
    for i in range(len(list_input)//4 * device_num, len(list_input)//4 * (device_num+1)):
        img_input, img_gt = list_input[i] / scale, list_gt[i] / scale
        
        img_input = torch.from_numpy(img_input.astype(np.float32)).permute(2,0,1).cuda(device_num)
        img_gt = torch.from_numpy(img_gt.astype(np.float32)).permute(2,0,1).cuda(device_num)

        mse_ori, fft_abs_ori, fft_angle_ori = calc_losses(_lambda, img_gt, img_input)
        loss_total_ori = mse_ori * _lambda[0] + fft_abs_ori * _lambda[1] + fft_angle_ori * _lambda[2]
        
        orig_h, orig_w = img_gt.shape[1], img_gt.shape[2]
        st_h, st_w = int( (orig_h-target_h) / 2), int( (orig_w-target_w) / 2) 

        warning_shift_value(st_h, st_w, max_shift)
    
        img_gt_crop = img_gt[:, st_h:st_h+target_h, st_w:st_w+target_w]
        img_gt_crop = img_gt_crop.cuda(device_num)
        loss_min, loss_total = 100000000, -1
        for j in range(max_shift*2):
            for k in range(max_shift*2):
                img_input_crop = img_input[:, st_h-max_shift+j : st_h-max_shift+j+target_h, st_w-max_shift+k : st_w-max_shift+k+target_w]
                img_input_crop = img_input_crop
                mse, fft_abs, fft_angle = calc_losses(_lambda, img_gt_crop, img_input_crop)
                loss_total = mse * _lambda[0] + fft_abs * _lambda[1] + fft_angle * _lambda[2]
    
                if loss_total < loss_min:
                    loss_min = loss_total
                    offset_h, offset_w = j, k
                    img_input_res = img_input_crop   

        print(f' Loss of %s: %0.4f ===> %0.4f, where offsets for h and w are  %d, %d, respectively.' 
           % (fname_gt[i], loss_total_ori, loss_min, offset_h-max_shift, offset_w-max_shift) )
        
        log_message = f' Loss of %s: %0.4f ===> %0.4f, where offsets for h and w are  %d, %d, respectively.\n' % (fname_gt[i], loss_total_ori, loss_min, offset_h-max_shift, offset_w-max_shift)
        write_log(log_file_path, log_message)

        img_input_save = (img_input_res.cpu().numpy().squeeze()*scale) 
        img_gt_save = (img_gt_crop.cpu().numpy().squeeze()*scale) 
        
        save_png(img_input_save, f"{output_dir}/{fname_input[i].split('.')[0]}.png") # input
        save_png(img_gt_save, f"{output_dir}/{fname_gt[i].split('.')[0]}.png") # GT
    
    # print("\nDone.")


def run_alignment(input_dir, output_dir, target_h, target_w, max_shift, _lambda, img_intensity):

    # cam_1: GT, cam_0: input
    input_dir_0 = os.path.join(input_dir, 'cam_0/')
    input_dir_1 = os.path.join(input_dir, 'cam_1/')

    list_input = glob.glob(input_dir_1 + "*.png")
    list_gt = glob.glob(input_dir_0 + "*.png")
    print("input size = ", len(list_input), "gt_size = ",len(list_gt))
  
    list_input, fname_input = load_png(list_input)
    list_gt, fname_gt = load_png(list_gt)

    video_num = input_dir.split('/')[2]
    log_file_path = f'./logs/alignment/log_'+video_num+'.txt'

    print('-----'*10)
    print('Input dir 1:', input_dir_0)
    print('Input dir 2:', input_dir_1)
    print('Output dir:', output_dir)
    print('Target H = %d' % target_h)
    print('Target W = %d' % target_w)
    print('Max shift = %d' % max_shift)
    print('-----'*10)

    data=[]
    data.append([list_input, list_gt, fname_input, fname_gt, output_dir, target_h, target_w, max_shift, _lambda, img_intensity, log_file_path, 0])
    data.append([list_input, list_gt, fname_input, fname_gt, output_dir, target_h, target_w, max_shift, _lambda, img_intensity, log_file_path, 1])
    data.append([list_input, list_gt, fname_input, fname_gt, output_dir, target_h, target_w, max_shift, _lambda, img_intensity, log_file_path, 2])
    data.append([list_input, list_gt, fname_input, fname_gt, output_dir, target_h, target_w, max_shift, _lambda, img_intensity, log_file_path, 3])
   
    with Pool(4) as p:
        p.map(pre_processing, data)

