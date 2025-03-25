"""
Trains a FastDVDnet model.
"""
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models import FastDVDnet
from dataset import ValDataset
# from dataloaders import train_dali_loader
from utils import svd_orthogonalization, close_logger, init_logging, normalize_augment
from train_common import resume_training, lr_scheduler, log_train_psnr, \
					validate_and_log, save_model_checkpoint


from data.select_dataset import define_Dataset
from data.dataset_video_train_4vit import VideoRecurrentTrainDataset4Vit
from data.dataset_video_validation_4vit import VideoRecurrentValDataset4Vit

from torch.utils.data import DataLoader

import os
from utils import variable_to_cv2_image

dataset_opt_ik = {
    "train": {
            "name": "train_dataset",
            "dataset_type": "VideoRecurrentTrainDataset",
            "dataroot_gt": "UDC-VIT_npy/training/GT", #path
            "dataroot_lq": "UDC-VIT_npy/training/Input", #path
            "filename_tmpl": "04d",
            "filename_ext": "npy",
            "test_mode": False,
            "io_backend": {"type": "npy"},
            "num_frame": 5,
            "gt_size": 256,
            "interval_list": [1],
            "random_reverse": False,
            "use_hflip": False,
			"use_vflip": True,
            "use_rot": True,
            "dataloader_shuffle": True,
            "dataloader_num_workers": 36,
            "dataloader_batch_size": 64
        },

    "validation": {
		"name": "valid_dataset",
		"dataset_type": "VideoRecurrentTrainDataset",
		"dataroot_gt": "UDC-VIT_npy/validation/GT", #path
		"dataroot_lq": "UDC-VIT_npy/validation/Input", #path
		"filename_tmpl": "04d",
		"filename_ext": "npy",
		"test_mode": False,
		"io_backend": {"type": "npy"},
		"num_frame": 5,
		"gt_size": 256,
		"interval_list": [1],
		"use_rot": True,
		"dataloader_num_workers": 36,
		"dataloader_batch_size": 64
    	},
}

def main(**args):
	r"""Performs the main training loop
	"""

	# Init loggers
	writer, logger = init_logging(args)

	# Load dataset
	logger.info('> Loading datasets ...')
	
	train_set = VideoRecurrentTrainDataset4Vit(dataset_opt_ik["train"])
	loader_train = DataLoader(train_set,
								batch_size=dataset_opt_ik["train"]["dataloader_batch_size"], 
								shuffle=True,
								num_workers=dataset_opt_ik["train"]["dataloader_num_workers"],
								drop_last=True,
								pin_memory=True)
	
	valid_set = VideoRecurrentValDataset4Vit(dataset_opt_ik["validation"])
	dataset_val = DataLoader(valid_set, 
						  		batch_size=dataset_opt_ik["validation"]["dataloader_batch_size"],
								shuffle=False, 
								num_workers=dataset_opt_ik["validation"]["dataloader_num_workers"],
								drop_last=False, 
								pin_memory=True)

	num_minibatches = int(len(train_set) // dataset_opt_ik["train"]["dataloader_batch_size"])

	# Define GPU devices
	torch.backends.cudnn.benchmark = True # CUDNN optimization

	# Create model
	model = FastDVDnet()
	model = model.cuda()
	model = nn.DataParallel(model)
	

	# Define loss
	criterion = nn.MSELoss(reduction='sum')
	criterion.cuda()

	# Optimizer
	optimizer = optim.Adam(model.parameters(), lr=args['lr'])

	# Resume training or start anew
	start_epoch, training_params = resume_training(args, model, optimizer)

	# Training
	start_time = time.time()
	for epoch in range(start_epoch, args['epochs']):
		# Set learning rate
		current_lr, reset_orthog = lr_scheduler(epoch, args)
		
		if reset_orthog:
			training_params['no_orthog'] = True

		# set learning rate in optimizer
		for param_group in optimizer.param_groups:
			param_group["lr"] = current_lr

		logger.info('\nlearning rate %f' % current_lr)

		# train
		for i, data in enumerate(loader_train, 0):
			model.train()
			optimizer.zero_grad()
			imgn_train, gt_train = data['L'], data['H']
			N, _, H, W = imgn_train.size()

			# std dev of each sequence
			stdn = torch.empty((N, 1, 1, 1)).cuda().uniform_(args['noise_ival'][0], to=args['noise_ival'][1])

			# Send tensors to GPU
			gt_train = gt_train.cuda(non_blocking=True)
			imgn_train = imgn_train.cuda(non_blocking=True)
			noise_map = stdn.expand((N, 1, H, W)).cuda(non_blocking=True) # one channel per image

			# Evaluate model and optimize it
			out_train = model(imgn_train, noise_map)

			# Compute loss
			loss = criterion(gt_train, out_train) / (N*2)
			loss.backward()
			optimizer.step()

			# Results
			if training_params['step'] % args['save_every'] == 0:
				# Apply regularization by orthogonalizing filters
				if not training_params['no_orthog']:
					model.apply(svd_orthogonalization)

				# Compute training PSNR
				log_train_psnr(out_train, gt_train, loss, writer, logger, epoch, i, num_minibatches, training_params)

			# update step counter
			training_params['step'] += 1

		# save model and checkpoint
		training_params['start_epoch'] = epoch + 1
		save_model_checkpoint(model, args, optimizer, training_params, epoch)

		model.eval()

		# Validation and log images
		validate_and_log(model_temp=model, dataset_val=dataset_val, valnoisestd=args['val_noiseL'], temp_psz=args['temp_patch_size'], writer=writer, epoch=epoch, lr=current_lr, logger=logger, trainimg=imgn_train, out_dir=args['log_dir']+'/val/'+str(epoch), batchsize=args['batch_size'])

	# Print elapsed time
	elapsed_time = time.time() - start_time
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

	# Close logger file
	close_logger(logger)



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train the denoiser")

	#Training parameters
	parser.add_argument("--batch_size", type=int, default=64, 	\
					 help="Training batch size")
	parser.add_argument("--epochs", "--e", type=int, default=80, \
					 help="Number of total training epochs")
	parser.add_argument("--resume_training", "--r", action='store_true',\
						help="resume training from a previous checkpoint")
	parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60], \
						help="When to decay learning rate; should be lower than 'epochs'")
	parser.add_argument("--lr", type=float, default=1e-3, \
					 help="Initial learning rate")
	parser.add_argument("--no_orthog", action='store_true',\
						help="Don't perform orthogonalization as regularization")
	parser.add_argument("--save_every", type=int, default=10,\
						help="Number of training steps to log psnr and perform \
						orthogonalization")
	parser.add_argument("--save_every_epochs", type=int, default=5,\
						help="Number of training epochs to save state")
	parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 55], \
					 help="Noise training interval")
	parser.add_argument("--val_noiseL", type=float, default=25, \
						help='noise level used on validation set')
	

	# Preprocessing parameters
	parser.add_argument("--patch_size", "--p", type=int, default=96, help="Patch size")
	parser.add_argument("--patch_size_val", "--pv", type=int, default=256, help="Patch size")
	parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
	parser.add_argument("--max_number_patches", "--m", type=int, default=256000, \
						help="Maximum number of patches")
	

	# Dirs
	parser.add_argument("--log_dir", type=str, default="logs", \
					 help='path of log files')
	parser.add_argument("--trainset_dir", type=str, default=None, \
					 help='path of trainset')
	parser.add_argument("--valset_dir", type=str, default=None, \
						 help='path of validation set')
	argspar = parser.parse_args()

	# Normalize noise between [0, 1]
	argspar.val_noiseL /= 255.
	argspar.noise_ival[0] /= 255.
	argspar.noise_ival[1] /= 255.

	print("\n### Training FastDVDnet denoiser model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	
	dataset_opt_ik["train"]["dataroot_gt"] = os.path.join(argspar.trainset_dir, 'GT')
	dataset_opt_ik["train"]["dataroot_lq"] = os.path.join(argspar.trainset_dir, 'Input')
	dataset_opt_ik["train"]["gt_size"] = argspar.patch_size
	dataset_opt_ik["train"]["num_frame"] = argspar.temp_patch_size

	dataset_opt_ik["validation"]["dataroot_gt"] = os.path.join(argspar.valset_dir, 'GT')
	dataset_opt_ik["validation"]["dataroot_lq"] = os.path.join(argspar.valset_dir, 'Input')
	dataset_opt_ik["validation"]["gt_size"] = argspar.patch_size_val
	dataset_opt_ik["validation"]["num_frame"] = argspar.temp_patch_size
	
	dataset_opt_ik["train"]["dataloader_batch_size"] = argspar.batch_size
	dataset_opt_ik["validation"]["dataloader_batch_size"] = argspar.batch_size

	main(**vars(argspar))
