# import os
import sys
# import natsort
# import glob
# from PIL import Image
# import cv2
# import imageio.v2 as imageio
# import numpy as np
# import torch
# import rawpy as rp

from check_fps import *
from yuv2png import *
from alignment import *
from wt_align_list import *
#from get_pck import *
from get_pck_mp import *
import shutil

def move_png_files(input_dir, output_dir, video_num):

    cam0_output_dir = os.path.join(output_dir, 'Input', video_num)
    cam1_output_dir = os.path.join(output_dir, 'GT', video_num)

    if not os.path.exists(cam0_output_dir):
        os.makedirs(cam0_output_dir)
    if not os.path.exists(cam1_output_dir):
        os.makedirs(cam1_output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.startswith('cam_0'):
            src_path = os.path.join(input_dir, file_name)
            dst_path = os.path.join(cam0_output_dir, file_name)
            shutil.move(src_path, dst_path)
        elif file_name.startswith('cam_1'):
            src_path = os.path.join(input_dir, file_name)
            dst_path = os.path.join(cam1_output_dir, file_name)
            shutil.move(src_path, dst_path)


if __name__ == "__main__":

    min_dir = int(sys.argv[1])
    max_dir = int(sys.argv[2])
    target_h = int(sys.argv[3])
    target_w = int(sys.argv[4])
    max_shift = int(sys.argv[5]) # It is generally smaller than 10 with consistent values. If not, do PHYSICAL ALIGNMENT again.
    _lambda = [int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])] # mse, fft_abs, fft_phase
    img_intensity = int(sys.argv[9])
    RUN_CONVERT = True if int(sys.argv[10]) == 1 else False
    RUN_ALIGN = True if int(sys.argv[11]) == 1 else False
    RUN_PCK = True if int(sys.argv[12]) == 1 else False

    print("====="*10)
    print("Begin running the processes below:")
    if RUN_CONVERT:
        print("Converting yuv to png.")
    if RUN_ALIGN:
        print("Alignment.")
    if RUN_PCK:
        print("Calculating PCK.")
    print("====="*10)

    # Run the process
    min_dir, max_dir = find_nearest_folder(min_dir, max_dir)
    check_fps(min_dir, max_dir)

    for folder_num in range(min_dir, max_dir + 1):
        print("Running for the directory name:", str(folder_num) + ".")
        
        if RUN_CONVERT:
            print("Converting yuv to png......")

            input_dir = os.path.join('./yuv/', str(folder_num))
            output_dir = os.path.join('./png/', str(folder_num))
            if os.path.exists(input_dir):
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                run_yuv2png(input_dir, output_dir, 30)#220)
            
            print("Converting finished.\n")

        if RUN_ALIGN:
            print("Alignment begins......")

            input_dir = os.path.join('./png/', str(folder_num))
            output_dir = os.path.join('./png_aligned/', str(folder_num))
            if os.path.exists(input_dir):
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)            
                run_alignment(input_dir, output_dir, target_h, target_w, max_shift, _lambda, img_intensity)
            
            input_dir = os.path.join('./png_aligned/', str(folder_num))
            output_dir = os.path.join('./logs/alignment_list/', 'udc_'+str(folder_num)+'.csv')
            write_aligned_list(input_dir, output_dir)

        if RUN_PCK:
            input_dir = os.path.join('./png_aligned/', str(folder_num)+'/')
            list_dir = os.path.join('./logs/alignment_list/', 'udc_'+str(folder_num)+'.csv')
            run_calc_pck(input_dir, list_dir)
        
        input_dir = os.path.join('./png_aligned/', str(folder_num))
        output_dir = 'udc_vit/UDC-VIT/'
        move_png_files(input_dir, output_dir, str(folder_num))
    
    print("\nFinished all processes: yuv2png, alignment, and calc_pck.")
