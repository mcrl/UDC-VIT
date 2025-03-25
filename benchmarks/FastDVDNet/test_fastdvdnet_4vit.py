#!/bin/sh
"""
Denoise all the sequences existent in a given folder using FastDVDnet.
"""
import os
import argparse
import time
import cv2
import torch
import torch.nn as nn
from models import FastDVDnet
from fastdvdnet import denoise_seq_fastdvdnet
from utils import batch_psnr, init_logger_test, \
				variable_to_cv2_image, remove_dataparallel_wrapper, open_sequence, close_logger

NUM_IN_FR_EXT = 5 # temporal size of patch

import numpy as np
from torch.utils.data import DataLoader
from data.dataset_video_test_4vit import VideoRecurrentTestDataset4Vit
from fastdvdnet import denoise_seq_fastdvdnet_ik
from skimage.metrics.simple_metrics import peak_signal_noise_ratio

dataset_opt_ik = {
	"test": {
		"name": "test_dataset",
		"dataset_type": "VideoRecurrentTrainDataset",
		"dataroot_gt": "UDC-VIT_npy/test/GT", #path
		"dataroot_lq": "UDC-VIT_npy/test/Input", #path
		"filename_tmpl": "04d",
		"filename_ext": "npy",
		"test_mode": False,
		"io_backend": {"type": "npy"},
		"num_frame": 5,
		"interval_list": [1],
		"use_rot": True,
		"dataloader_num_workers": 36,
		"dataloader_batch_size": 64
    	}
}

def test_fastdvdnet(**args):
	"""Denoises all sequences present in a given folder. Sequences must be stored as numbered
	image sequences. The different sequences must be stored in subfolders under the "test_path" folder.

	Inputs:
		args (dict) fields:
			"model_file": path to model
			"test_path": path to sequence to denoise
			"suffix": suffix to add to output name
			"max_num_fr_per_seq": max number of frames to load per sequence
			"noise_sigma": noise level used on test set
			"dont_save_results: if True, don't save output images
			"no_gpu": if True, run model on CPU
			"save_path": where to save outputs as png
			"gray": if True, perform denoising of grayscale images instead of RGB
	"""

	# If save_path does not exist, create it
	if not os.path.exists(args['save_path']):
		os.makedirs(args['save_path'])
	logger = init_logger_test(args['save_path'])

	# Sets data type according to CPU or GPU modes
	if args['cuda']:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	print('torch.cuda.device_count() = ', torch.cuda.device_count())

	# Create models
	print('Loading models ...')
	model_temp = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)

	# Load saved weights
	state_temp_dict = torch.load(args['model_file'], map_location=device)
	
	if args['cuda']:
		model_temp = nn.DataParallel(model_temp).cuda()
	else:
		state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
	model_temp.load_state_dict(state_temp_dict['state_dict'])

	model_temp.eval()

	# process data
	test_set = VideoRecurrentTestDataset4Vit(dataset_opt_ik["test"])
	dataset_test = DataLoader(test_set, 
							batch_size=dataset_opt_ik["test"]["dataloader_batch_size"],
							shuffle=False, 
							num_workers=dataset_opt_ik["test"]["dataloader_num_workers"],
							drop_last=False, 
							pin_memory=True)
	psnr_test = 0
	batchcnt = 0
	
	outsave_path = os.path.join(args['save_path'], 'out')
	if not os.path.exists(outsave_path):
		os.makedirs(outsave_path)

	with torch.no_grad():
		for i, data in enumerate(dataset_test):

			imgn_train, gt_train = data['L'], data['H']
			N, _, H, W = imgn_train.size()

			if N == args["batch_size"]:
				stdn = torch.empty((N, 1, 1, 1)).cuda().fill_(args["noise_sigma"])

				gt_train = gt_train.cuda(non_blocking=True)
				imgn_train = imgn_train.cuda(non_blocking=True)
				noise_map = stdn.expand((N, 1, H, W)).cuda(non_blocking=True) # one channel per image

				out_val = model_temp(imgn_train, noise_map)

				psnr_now = batch_psnr(torch.clamp(out_val, 0., 1.), gt_train, 1.)
				psnr_test += psnr_now
				batchcnt = batchcnt + 1

				outsave_path_folder = os.path.join(args['save_path'], 'out', data['H_folder'][0])
				if not os.path.exists(outsave_path_folder):
					os.makedirs(outsave_path_folder)

				svimg_out = variable_to_cv2_image(out_val.squeeze(0).clamp(0., 1.))
				svimg_out_fname = os.path.join(outsave_path_folder, data['H_filename'][0] + '.png')
				cv2.imwrite(svimg_out_fname, svimg_out)

				logger.info("[Image %d / %d] PSNR: %.4f / File path: %s" % (batchcnt, len(dataset_test), psnr_now, svimg_out_fname))

		psnr_test /= batchcnt
		logger.info("[Test Result] Average PSNR: %.4f" % (psnr_test))
		
	logger.info("Finished denoising {} : Tested images= {}".format(args['test_path'], len(dataset_test)))
	logger.info("Finished denoising : Test PSNR = {:.4f} dB".format(psnr_test))

	# close logger
	close_logger(logger)


if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet")
	parser.add_argument("--model_file", type=str, default="./model.pth", help='path to model of the pretrained denoiser')
	parser.add_argument("--test_path", type=str, default="./data/rgb/Kodak24", help='path to sequence to denoise')
	parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
	parser.add_argument("--noise_sigma", type=float, default=0, help='noise level used on test set')
	parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
	parser.add_argument("--save_path", type=str, default='./results', help='where to save outputs as png')
	parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
	parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
	
	argspar = parser.parse_args()

	argspar.noise_sigma /= 255.
	argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

	print("\n### Testing FastDVDnet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	dataset_opt_ik["test"]["dataroot_gt"] = os.path.join(argspar.test_path, 'GT')
	dataset_opt_ik["test"]["dataroot_lq"] = os.path.join(argspar.test_path, 'Input')

	dataset_opt_ik["test"]["dataloader_batch_size"] = argspar.batch_size

	test_fastdvdnet(**vars(argspar))
