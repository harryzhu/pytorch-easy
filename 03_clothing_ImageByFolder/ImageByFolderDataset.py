#!/usr/bin/env python
# coding: utf-8

import os
import time
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torch.utils.data import ConcatDataset

from torchvision import transforms
import matplotlib.pyplot as plt

import config as CFG 

transform = transforms.Compose([
	transforms.Resize([CFG.image_resize_width,CFG.image_resize_height]),
	transforms.ToTensor(),
	transforms.Normalize(CFG.transform_normalization_mean, CFG.transform_normalization_std),
	])

class ImageByFolderDataset(data.Dataset):
	def __init__(self, images, labels, transform):
		self.images = images
		self.labels = labels
		self.transforms = transform

	def __getitem__(self, idx):
		image = self.images[idx]
		label = self.labels[idx]
		pil_image = Image.open(image)
		pil_image = pil_image.convert('RGB')
		#pil_image = pil_image.convert('L')
		data = self.transforms(pil_image)

		return data, label

	def __len__(self):
		return len(self.images)

def load_image_classes():
	c = []
	with open(CFG.image_classes_file,'r') as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip()
			if len(line) > 0:
				c.append(line)
	return c

def gen_image_classes():
	classes_list = []
	with open(CFG.image_classes_file,"w") as cf:
		all_dirs = glob.glob(CFG.train_set_dir+"/*")
		for d in all_dirs:
			if os.path.isdir(d):
				classes_list.append(os.path.basename(d))
		classes_list = set(classes_list)
		classes_list = sorted(classes_list)
		#print("gen_image_classes:",classes_list)	
		cf.writelines("\r\n".join(classes_list))	


def images2labels(images, class_list):
	labels = []
	for image in images:
		labels.append(class_list.index(str(os.path.basename(os.path.dirname(image)))))
	return labels

def image_normalize_mean_std(sets=[]):
	if len(sets) == 0:
		return None
	x2 = torch.stack([sample[0] for sample in ConcatDataset(sets)])

	mean = torch.mean(x2, dim=(0,2,3))
	#print("mean:", mean)

	std = torch.std(x2, dim=(0,2,3))
	#print("std:", std)

	return mean, std

		
all_train_images = glob.glob(CFG.train_set_dir + '/**/*'+ CFG.image_extension)
all_test_images = glob.glob(CFG.test_set_dir + '/**/*'+ CFG.image_extension)


if not os.path.exists(CFG.image_classes_file):
	gen_image_classes()

all_classes = load_image_classes()
CFG.num_classes = len(all_classes)
print(f"all_classes: {CFG.num_classes}: {all_classes}")

all_train_labels = images2labels(all_train_images,all_classes)
all_test_labels = images2labels(all_test_images,all_classes)

#print("all_train_labels:",all_train_labels)
#print("all_test_labels:",all_test_labels)

dataset_train = ImageByFolderDataset(all_train_images,all_train_labels, transform)
dataset_test = ImageByFolderDataset(all_test_images,all_test_labels, transform)

print(f'train images: {len(dataset_train)}')
print(f'test images: {len(dataset_test)}')

if not os.path.exists(CFG.transform_normalization_file):
	normalization_mean, normalization_std = image_normalize_mean_std([dataset_train, ])
	res = f'normalization_mean: {normalization_mean}, \nnormalization_std: {normalization_std}'
	with open(CFG.transform_normalization_file,'w') as f:
		f.write(res)
	print(res)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = CFG.batch_size, shuffle = True)
# images_batch, labels_batch = next(iter(dataloader_train))
# print("images_batch.shape: ",images_batch.shape)

dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = CFG.batch_size, shuffle = True)
# images_batch, labels_batch = next(iter(dataloader_test))
# print(images_batch.shape)


# plt.figure(figsize=(64, 64))
# for i, (img, label) in enumerate(zip(images_batch[:12], labels_batch[:12])):
# 	img = img.permute(1, 2, 0).numpy()
# 	plt.subplot(3, 4, i+1)
# 	plt.title(label.item())
# 	plt.imshow(img)
# plt.show()


