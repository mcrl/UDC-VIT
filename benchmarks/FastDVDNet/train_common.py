"""
Different common functions for training the models.
"""
import os
import time
import torch
import torchvision.utils as tutils
from utils import batch_psnr
from fastdvdnet import denoise_seq_fastdvdnet, denoise_seq_fastdvdnet_ik


import cv2
from utils import variable_to_cv2_image
import numpy as np
from skimage.metrics.simple_metrics import peak_signal_noise_ratio


def	resume_training(argdict, model, optimizer):
	""" Resumes previous training or starts anew
	"""
	if argdict['resume_training']:
		resumef = os.path.join(argdict['log_dir'], 'ckpt.pth')
		if os.path.isfile(resumef):
			checkpoint = torch.load(resumef)
			print("> Resuming previous training")
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			new_epoch = argdict['epochs']
			new_milestone = argdict['milestone']
			current_lr = argdict['lr']
			argdict = checkpoint['args']
			training_params = checkpoint['training_params']
			start_epoch = training_params['start_epoch']
			argdict['epochs'] = new_epoch
			argdict['milestone'] = new_milestone
			argdict['lr'] = current_lr
			print("=> loaded checkpoint '{}' (epoch {})"\
				  .format(resumef, start_epoch))
			print("=> loaded parameters :")
			print("==> checkpoint['optimizer']['param_groups']")
			print("\t{}".format(checkpoint['optimizer']['param_groups']))
			print("==> checkpoint['training_params']")
			for k in checkpoint['training_params']:
				print("\t{}, {}".format(k, checkpoint['training_params'][k]))
			argpri = checkpoint['args']
			print("==> checkpoint['args']")
			for k in argpri:
				print("\t{}, {}".format(k, argpri[k]))

			argdict['resume_training'] = False
		else:
			raise Exception("Couldn't resume training with checkpoint {}".\
				   format(resumef))
	else:
		start_epoch = 0
		training_params = {}
		training_params['step'] = 0
		training_params['current_lr'] = 0
		training_params['no_orthog'] = argdict['no_orthog']

	return start_epoch, training_params

def lr_scheduler(epoch, argdict):
	"""Returns the learning rate value depending on the actual epoch number
	By default, the training starts with a learning rate equal to 1e-3 (--lr).
	After the number of epochs surpasses the first milestone (--milestone), the
	lr gets divided by 100. Up until this point, the orthogonalization technique
	is performed (--no_orthog to set it off).
	"""
	# Learning rate value scheduling according to argdict['milestone']
	reset_orthog = False
	if epoch > argdict['milestone'][1]:
		current_lr = argdict['lr'] / 1000.
		reset_orthog = True
	elif epoch > argdict['milestone'][0]:
		current_lr = argdict['lr'] / 10.
	else:
		current_lr = argdict['lr']
	return current_lr, reset_orthog

def	log_train_psnr(result, imsource, loss, writer, logger, epoch, idx, num_minibatches, training_params):
	'''Logs trai loss.
	'''
	#Compute pnsr of the whole batch
	psnr_train = batch_psnr(torch.clamp(result, 0., 1.), imsource, 1.)

	# Log the scalar values
	writer.add_scalar('loss', loss.item(), training_params['step'])
	writer.add_scalar('PSNR on training data', psnr_train, training_params['step'])

	logger.info("[epoch %d][%d/%d] loss: %f / PSNR_train: %f" % (epoch+1, idx+1, num_minibatches, loss.item(), psnr_train))

def save_model_checkpoint(model, argdict, optimizer, train_pars, epoch):
	"""Stores the model parameters under 'argdict['log_dir'] + '/net.pth'
	Also saves a checkpoint under 'argdict['log_dir'] + '/ckpt.pth'
	"""
	torch.save(model.state_dict(), os.path.join(argdict['log_dir'], 'net.pth'))
	save_dict = { \
		'state_dict': model.state_dict(), \
		'optimizer' : optimizer.state_dict(), \
		'training_params': train_pars, \
		'args': argdict\
		}
	torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckpt.pth'))

	if epoch % argdict['save_every_epochs'] == 0:
		torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckpt_e{}.pth'.format(epoch+1)))
	del save_dict

def validate_and_log(model_temp, dataset_val, valnoisestd, temp_psz, writer, epoch, lr, logger, trainimg, out_dir, batchsize):
	"""Validation step after the epoch finished
	"""
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
		
	t1 = time.time()
	psnr_val = 0

	batchcnt = 0
	with torch.no_grad():	
		for i, data in enumerate(dataset_val):

			imgn_train, gt_train = data['L'], data['H']
			N, _, H, W = imgn_train.size()

			if N == batchsize:
				stdn = torch.empty((N, 1, 1, 1)).cuda().fill_(valnoisestd)

				gt_train = gt_train.cuda(non_blocking=True)
				imgn_train = imgn_train.cuda(non_blocking=True)
				noise_map = stdn.expand((N, 1, H, W)).cuda(non_blocking=True) # one channel per image

				out_val = model_temp(imgn_train, noise_map)
				

				psnr_now = batch_psnr(torch.clamp(out_val, 0., 1.), gt_train, 1.)

				psnr_val += psnr_now

				batchcnt = batchcnt + 1
			
		psnr_val /= batchcnt
		
		
		t2 = time.time()

		logger.info("[epoch %d] PSNR_val: %.4f, on %.2f sec" % (epoch+1, psnr_val, (t2-t1)))
		writer.add_scalar('PSNR on validation data', psnr_val, epoch)
		writer.add_scalar('Learning rate', lr, epoch)

		
