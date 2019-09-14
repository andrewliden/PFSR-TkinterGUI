#Some of these imports could probably be snipped
#Imports from the original eval.py
import torch
from torch import optim, nn
import argparse
from dataloader import CelebDataSet
from torch.utils.data import DataLoader
from model import Generator
import os
from torch.autograd import Variable, grad
import sys
from torchvision import utils
from math import log10
from ssim import ssim, msssim

#Imports from dataloader.py
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from os.path import join
from PIL import Image

class simpleData(Dataset):
	"""
	Simple Data:
	intended to provide similar functionality to the bundled CelebA dataset, 
	but for the use case of testing only, on single images, using a pre-trained network,
	rather than on a large dataset, with a mix of training and testing.
	
	Parameters:
        img_path (str)     -- Generic path to a single image
	
	Also similarly, the goal is as stated:
	    Center crop the aligned image to 178x178 to include the face area and then downsample to 128x128(Step3).
    In addition, for progressive training, the target image for each step is resized to 32x32(Step1) and 64x64(Step2).
		
	"""
	
	#If no image is specified, it is assumed that the image to be worked on is an image called img.jpg in the same directory as the script.
	def __init__(self, img_path = './img.jpg'):
		#This dataset object is intended for testing only.
		self.state = 'test'

		#This is still used for compatibility reasons:
		#	Using the test image list requires very few code changes.
		test_img_list = []
		test_img_list.append(img_path)
		self.image_list = test_img_list
		#image center cropping
		self.pre_process = transforms.Compose([
									transforms.CenterCrop((178, 178)),
									transforms.Resize((128, 128)),
									])

		self.totensor = transforms.Compose([
									transforms.ToTensor(),
									transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
									])

		self._64x64_down_sampling = transforms.Resize((64, 64))
		self._32x32_down_sampling = transforms.Resize((32, 32))
		self._16x16_down_sampling = transforms.Resize((16,16))
	
	def __getitem__(self, index):
		image_path = self.image_list[0]
		target_image = Image.open(image_path).convert('RGB')
		target_image = self.pre_process(target_image)
		x4_target_image = self._64x64_down_sampling(target_image)
		x2_target_image = self._32x32_down_sampling(x4_target_image)
		input_image = self._16x16_down_sampling(x2_target_image)
		
		x2_target_image = self.totensor(x2_target_image)
		x4_target_image = self.totensor(x4_target_image)
		target_image = self.totensor(target_image)
		input_image = self.totensor(input_image)
		
		return x2_target_image, x4_target_image, target_image, input_image

	def __len__(self):
		return len(self.image_list)
	

def simpleTest(device, dataloader, generator, MSE_Loss, step, alpha, resultpath = 'result.jpg'):
	for i, (x2_target_image, x4_target_image, target_image, input_image) in enumerate(dataloader):
		if step==1:
			target_image = x2_target_image.to(device)
		elif step==2:
			target_image = x4_target_image.to(device)
		else:
			target_image = target_image.to(device)

		input_image = input_image.to(device)
		predicted_image = generator(input_image, step, alpha)
		mse_loss = MSE_Loss(0.5*predicted_image+0.5, 0.5*target_image+0.5)
		psnr = 10*log10(1./mse_loss.item())
		_ssim = ssim(0.5*predicted_image+0.5, 0.5*target_image+0.5)
		ms_ssim = msssim(0.5*predicted_image+0.5, 0.5*target_image+0.5)

		sys.stdout.write('\r [%d/%d] Test progress... PSNR: %6.4f'%(i, len(dataloader), psnr))
		utils.save_image(0.5*predicted_image+0.5, resultpath)
	print('Image generated!')

def testImage(imagePath='img.jpg', workers=4, resultPath='result.jpg'):
	#Create a torch device using CUDA if available, otherwise use the CPU
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#Create a dataset object of the "simpleTest" type
	dataset = simpleData(imagePath)
	#This is a single image demo.  
	BATCHSIZE = 1
	#Create a data loader.
	dataloader = DataLoader(dataset=dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=workers, pin_memory=True)

	#create a generator
	generator = Generator().to(device)
	#Assume the checkpoint path is the default.
	CHECKPOINT_PATH = './checkpoints/generator_checkpoint.ckpt'
	g_checkpoint = torch.load(CHECKPOINT_PATH)

	generator.load_state_dict(g_checkpoint['model_state_dict'], strict=False)
	step = g_checkpoint['step']
	alpha = g_checkpoint['alpha']
	iteration = g_checkpoint['iteration']

	print('Pre-trained model is loaded step:%d, iteration:%d'%(step, iteration))
	MSE_Loss = nn.MSELoss()

	simpleTest(device, dataloader, generator, MSE_Loss, step, alpha, resultPath)
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser('Simple, single image implementation of Progressive Face Super-Resolution Attention to Face Landmarks')
	parser.add_argument('--img-path', default='./img.jpg', type=str)
	parser.add_argument('--workers', default=4, type=int)
	parser.add_argument('--result-path', default='./result.jpg', type=str)
	args = parser.parse_args()
	testImage(args.img_path, args.workers, args.result_path)