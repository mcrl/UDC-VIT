import os
import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

import kornia_moons
from kornia_moons.feature import *
from PIL import Image

import torch
import torch.cuda
import torchvision
from PIL import Image
import torchvision.transforms as transforms


#def load_torch_image(fname):
#    img = K.image_to_tensor(cv2.imread(fname), False).float() /255.
#    img = K.color.bgr_to_rgb(img)
#    return img


def load_torch_image(fname, device):
    img = Image.open(fname).convert("RGB")
    width, height = img.size
    MAX_SIZE = max(width, height)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0).to(device)
    return img, MAX_SIZE


def match_and_draw_gpu(im_path, img_in1, img_in2):
    DRAW = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img1, MAX_SIZE1 = load_torch_image(im_path + img_in1, device)
    img2, MAX_SIZE2 = load_torch_image(im_path + img_in2, device)

    MAX_SIZE = MAX_SIZE1 if MAX_SIZE1 == MAX_SIZE2 else sys.exit("Error: Image sizes are different.")
    
    matcher = KF.LoFTR(pretrained='outdoor')
    matcher = torch.nn.DataParallel(matcher)
    matcher = matcher.to(device)

    input_dict = {
        "image0": K.color.rgb_to_grayscale(img1).to(device), # Convert to grayscale and send to GPU
        "image1": K.color.rgb_to_grayscale(img2).to(device)  # Convert to grayscale and send to GPU
    }

    with torch.no_grad():
        correspondences = matcher(input_dict)

    if DRAW:
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()

        H, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        inliers = inliers > 0

        kornia_moons.feature.draw_LAF_matches(
            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1, -1, 2),
                                        torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                                        torch.ones(mkpts0.shape[0]).view(1, -1, 1)), 

            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1, -1, 2),
                                        torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                                        torch.ones(mkpts1.shape[0]).view(1, -1, 1)), 

            torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
            K.tensor_to_image(img1),
            K.tensor_to_image(img2),
            inliers,
            draw_dict={'inlier_color': (0.2, 1, 0.2),
                    'tentative_color': None,
                    'feature_color': (0.2, 0.5, 1), 'vertical': False}
        )
    
    return correspondences, MAX_SIZE


def calculate_pck(correspondences, ind, pck_0p002_arr, pck_0p005_arr, pck_0p01_arr, pck_0p03_arr, pck_0p10_arr, MAX_SIZE, dirname):
    # Keypoint coordinates for last prediction - only for showing structure 
    key_gt = correspondences['keypoints0'].cpu().numpy().T
    key_in = correspondences['keypoints1'].cpu().numpy().T

    ind = int(ind)
    np.savetxt('./'+dirname+'/keypoints_'+str(ind)+'_gt.txt', key_gt, delimiter=',', fmt='%.1f')
    np.savetxt('./'+dirname+'/keypoints_'+str(ind)+'_input.txt', key_in, delimiter=',', fmt='%.6f')

    diff = np.abs(key_gt - key_in)
    diff = np.ndarray.flatten(diff)
    count_0p002, count_0p005, count_0p01, count_0p03, count_0p10 = 0, 0, 0, 0, 0

    for i in range(len(diff)):
        if diff[i] <= MAX_SIZE * 0.002:
            count_0p002 += 1
        if diff[i] <= MAX_SIZE * 0.005:
            count_0p005 += 1    
        if diff[i] <= MAX_SIZE * 0.01:
            count_0p01 += 1
        if diff[i] <= MAX_SIZE * 0.03:
            count_0p03 += 1
        if diff[i] <= MAX_SIZE * 0.10:
            count_0p10 += 1

    pck_0p002, pck_0p005, pck_0p01, pck_0p03, pck_0p10 = count_0p002 / len(diff), count_0p005 / len(diff), count_0p01 / len(diff), count_0p03 / len(diff), count_0p10 / len(diff)
    print(".......Each PCK:", pck_0p002, pck_0p005, pck_0p01, pck_0p03, pck_0p10)

    pck_0p002_arr.append(pck_0p002), pck_0p005_arr.append(pck_0p005), pck_0p01_arr.append(pck_0p01), pck_0p03_arr.append(pck_0p03), pck_0p10_arr.append(pck_0p10)
    print(".......AVG PCK:", np.mean(pck_0p002_arr), np.mean(pck_0p005_arr), np.mean(pck_0p01_arr), np.mean(pck_0p03_arr), np.mean(pck_0p10_arr))

    np.savetxt('./'+dirname+'/pck_0p002.txt', pck_0p002_arr, delimiter=',', fmt='%.4f')
    np.savetxt('./'+dirname+'/pck_0p005.txt', pck_0p005_arr, delimiter=',', fmt='%.4f')
    np.savetxt('./'+dirname+'/pck_0p01.txt', pck_0p01_arr, delimiter=',', fmt='%.4f')
    np.savetxt('./'+dirname+'/pck_0p03.txt', pck_0p03_arr, delimiter=',', fmt='%.4f')
    np.savetxt('./'+dirname+'/pck_0p10.txt', pck_0p10_arr, delimiter=',', fmt='%.4f')
    np.savetxt('./'+dirname+'/pck_avg_all.txt', [pck_0p002_arr, pck_0p005_arr, pck_0p01_arr, pck_0p03_arr, pck_0p10_arr], delimiter=',', fmt='%.4f')

    return np.mean(pck_0p002_arr), np.mean(pck_0p005_arr), np.mean(pck_0p01_arr), np.mean(pck_0p03_arr), np.mean(pck_0p10_arr) # pck_0p002_arr, pck_0p005_arr, pck_0p01_arr, pck_0p03_arr, pck_0p10_arr


def run_calc_pck():

    print("Calculating the percentage of correct keypoints (PCKs) between paired images.")

    im_path =  './png_aligned/'
    udc_dataset = pd.read_csv("./udc.csv")
    udc_dataset.species.value_counts().head(8)
    
    udc_list = udc_dataset["individual_id"].values.tolist()
    udc_list = list(set(udc_list))
    #global MAX_SIZE 
    #MAX_SIZE = 1920


    dirname = 'pck_res'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    
    pck_0p002_arr, pck_0p005_arr, pck_0p01_arr, pck_0p03_arr, pck_0p10_arr = [], [], [], [], []
    
    for i in udc_list:
        torch.cuda.empty_cache()
        query_str = "individual_id == " + str(i)
        img_to_draw = [file for file in udc_dataset.query(query_str).image]
    
        udc_1 = img_to_draw[1]
        udc_2 = img_to_draw[0]
        print(f'Matching: {udc_1} to {udc_2}')
    
        correspondences, MAX_SIZE = match_and_draw_gpu(im_path, udc_2, udc_1)
    
        pck_0p002, pck_0p005, pck_0p01, pck_0p03, pck_0p10 = calculate_pck(correspondences, i, pck_0p002_arr, pck_0p005_arr, pck_0p01_arr, pck_0p03_arr, pck_0p10_arr, MAX_SIZE, dirname)
    
    print("Finished calculating PCKs.\n")

    print("====="*50)
    distance = [round(threshold * MAX_SIZE, 1) for threshold in (0.002, 0.005, 0.01, 0.03, 0.10)]
    print("PCK for the distance:", distance)
    pcks = [round(pck * 100, 2) for pck in (pck_0p002, pck_0p005, pck_0p01, pck_0p03, pck_0p10)]
    print("PCKs:", pcks)
    print("====="*50)

    print("\nFinished all processes: yuv2png, alignment, and calc_pck.")
    torch.cuda.empty_cache()

