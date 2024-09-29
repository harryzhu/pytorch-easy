#!/usr/bin/env python
# coding: utf-8

import os
import time
import glob
import random
import numpy as np
from PIL import Image
import shutil

from typing import List, Tuple, Dict, AnyStr, KeysView, Any

import torch
from torch import nn
from torch.utils import data
from torchvision import transforms

import config as CFG 
from ImageByFolderDataset import *

import matplotlib.pyplot as plt
from tqdm import tqdm

class ImageByFolderNet(nn.Module):
	def __init__(self):
		super(ImageByFolderNet, self).__init__()
		self.conv0 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(5,5))
		self.conv1 = nn.Conv2d(8,24,kernel_size=(3,3))
		self.pool0 = nn.AvgPool2d(kernel_size=(5,5))
		self.pool1 = nn.AvgPool2d(kernel_size=(3,3))
		self.flatten = nn.Flatten()
		self.linear0 = nn.Linear(6144, 5200)
		self.linear1 = nn.Linear(5200,4000)
		self.linear2 = nn.Linear(4000,CFG.num_classes)
		self.relu = nn.ReLU()
		
		_layers = []
		_layers.append(self.conv0)
		_layers.append(self.pool0)
		_layers.append(self.conv1)
		_layers.append(self.pool1)
		_layers.append(self.flatten)
		_layers.append(self.linear0)
		_layers.append(self.relu)
		_layers.append(self.linear1)
		_layers.append(self.relu)
		_layers.append(self.linear2)
		_layers.append(self.relu)

		self.layers = _layers

	def forward(self,x):
		#return x
		#print("=== x ===",x.shape)
		output = self.conv0(x)
		output = self.pool0(output)
		#print("conv0:",output.shape)
		output = self.conv1(output)	
		output = self.pool1(output)
		#print("conv1:",output.shape)
		output = self.flatten(output)
		#print("flatten:",output.shape)
		# #print(output)
		output = self.linear0(output)
		#print("linear0:",output.shape)
		output = self.relu(output)
		output = self.linear1(output)
		#print("linear1:",output.shape)
		output = self.relu(output)
		output = self.linear2(output)
		#print("linear2:",output.shape)
		output = self.relu(output)
		#print("output:",output.shape)
		#print(output)
		return output


	def __len__(self):
		return len(self.layers)

	def __getitem__(self, item):
		return self.layers[item]	

	def __name__(self):
		return "ImageByFolderNet"


def train(net, loss_fn, train_iter, optimizer, epochs):
	accumulator = Accumulator(['train_loss','vali_loss','train_acc','vali_acc'])
	device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
	print('train on device:', device)
	net = net.to(device)
	epoch_start = 0
	model_path = None
	latest_pth = get_latest_file("output/*.pth")
	if latest_pth is not None:
		epoch_start = int(os.path.basename(latest_pth).replace("model_","").replace(".pth","").strip())
		model_path = latest_pth
	print(f'epoch_start: {epoch_start}')

	if model_path is not None:
		ckpt = torch.load(model_path, map_location=device)
		net.load_state_dict(ckpt)
		epochs = epochs + epoch_start
	#return None

	one_hot_f = nn.functional.one_hot
	epoch_loss = []
	last_loss = 0.0
	time_train_start = 0
	time_vali_start = 0
	for epoch in range(epoch_start,epochs):
		time_train_start = time.time()
		len_train = 0
		len_vali = 0

		net.train()
		epoch_loss.clear()
		correct_num = 0
		iter_batch = 0
		for img, label in train_iter:
			#print("label:",label)
			img,label = img.to(device,dtype=torch.float), label.to(device)
			oh_label = one_hot_f(label.long(), num_classes=CFG.num_classes)
			#print("one_hot_label:",oh_label)
			optimizer.zero_grad()
			#print(img.shape)
			y_hat = net(img)
			l = loss_fn(y_hat, oh_label.to(dtype=float))
			l.backward()
			optimizer.step()
			#
			epoch_loss.append(l.item())
			correct_num += (y_hat.argmax(dim=1, keepdim=True) == label.reshape(-1,1)).sum().item()
			len_train += len(label)

			accumulator['train_loss'].append(sum(epoch_loss)/len(epoch_loss))
			accumulator['train_acc'].append(correct_num/len_train)

			#print(f'{ iter_batch }: { iter_batch * CFG.batch_size }:')
			#print(':epoch_loss:',len(epoch_loss),epoch_loss)
			#print('correct_num:',correct_num)
			#print('len_train:',len_train)
		

		print(f'-----------epoch: {epoch+1} start --------------')
		print(f'epoch: {epoch+1} / {epochs} train loss: { accumulator["train_loss"][-1] }')
		print(f'epoch: {epoch+1} / {epochs} train acc: { accumulator["train_acc"][-1] }')
		print(f'epoch: {epoch+1} / {epochs} train time: {int(time.time()-time_train_start)} sec')

			# save
		if (((epoch+1) % 10 == 0) or (epoch + 1 >= epochs)):
			torch.save(net.state_dict(), './output/model_'+str(epoch+1)+'.pth')
			#torch.save(net.state_dict(), './output/model_current.pth')


def test(model, all_test_images):
	latest_pth = get_latest_file("output/*.pth")
	model_path = None
	if os.path.exists(latest_pth):
		model_path = latest_pth.replace('\\','/')

	if model_path is None:
		print("Error: cannot load any pth")
		return None
	else:
		print(f'loading model: {model_path}')

	device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
	model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
	model.eval()

	test_ok = 0.0
	test_error = 0.0
	random.shuffle(all_test_images)
	
	for image in all_test_images:
		if inference_image(image, model):
			test_ok += 1.0
		else:
			test_error +=1.0


	ratio = 0.0
	if test_ok + test_error > 0:
		ratio = test_ok / (test_ok + test_error)
		print(f"[TEST] acc : {ratio:>.4f}, {test_ok} / { (test_ok + test_error) }")

	txt_1 = f'[TEST] acc: {ratio:>.4f} <= {test_ok} / { (test_ok + test_error) } <= {model_path}\n'
	save_append("./output/test_acc.txt", txt_1 , 'a+')

def inference_image(image_file,model):
	should_class = os.path.basename(os.path.dirname(image_file))
	should_label = all_classes.index(should_class)
	#print(f'should_class:{should_class}, should_label: {should_label}')

	infer_images = ImageByFolderDataset([image_file,],[should_label,], transform)
	infer_dataloader = torch.utils.data.DataLoader(infer_images, batch_size=1,)
	images_infer_batch, labels_infer_batch = next(iter(infer_dataloader))
	#print(images_infer_batch.shape)

	output = None
	with torch.no_grad():
	    output= model(images_infer_batch)  # 得到推理结果
	#print(f'output: {output}')

	#pred = torch.argmax((output))
	pred = torch.argmax((output))
	#print("prediction: ",pred.item())
	class_idx = int(pred.item())
	if should_class == all_classes[class_idx]:
		#print(f"OK: {all_classes[class_idx]} <= {image_file}")
		infer_ok_path = "output/test_ok/" + should_class +"/" + os.path.basename(image_file)
		if not os.path.isdir(os.path.dirname(infer_ok_path)):
			os.makedirs(os.path.dirname(infer_ok_path))
		shutil.copy(image_file, infer_ok_path)
		return True
	else:
		#print(f"ERROR: should {should_class}, but {all_classes[class_idx]}, {image_file}")
		infer_error_path = "output/test_error/" + should_class +"/" + os.path.basename(image_file)
		if not os.path.isdir(os.path.dirname(infer_error_path)):
			os.makedirs(os.path.dirname(infer_error_path))
		shutil.copy(image_file, infer_error_path)
		return False


def save_append(fpath,data,mode='a'):
	if not os.path.isdir(os.path.dirname(fpath)):
		os.mkdir(os.path.dirname(fpath))
	with open(fpath, mode) as f:
		f.write(data)
		f.close()

def get_latest_file(pattern):
	flist = glob.glob(pattern)
	fmtime = 0
	fname = None
	for pth in flist:
		pth_mtime = os.path.getmtime(pth)
		if pth_mtime > fmtime:
			fmtime = pth_mtime
			fname = pth
	return fname



class Accumulator():
    """
    collecting metrics in experiment
    """
    def __init__(self, names: List[Any]):
        self.accumulator = {}
        if not isinstance(names, list):
            raise Exception(f'type error, expected list but got {type(names)}')
        for name in names:
            self.accumulator[name] = list()

    def __getitem__(self, item) -> List[Any]:
        if item not in self.accumulator.keys():
            raise Exception(f'key error, {item} is not in accumulator')
        return self.accumulator[item]

    def add(self, name: AnyStr, val: Any):
        self.accumulator[name].append(val)

    def add_name(self, name: AnyStr):
        if name in self.accumulator.keys():
            raise Exception(f'{name} is  already in accumulator.keys')
        self.accumulator[name] = list()

    def gets(self) -> Dict[AnyStr, Any]:
        return self.accumulator

    def get_item(self, name: AnyStr) -> List[Any]:
        if name not in self.accumulator.keys():
            raise Exception(f'key error, {name} is not in accumulator')
        return self.accumulator[name]

    def clear(self):
        self.accumulator.clear()

    def get_names(self) -> KeysView:
        return self.accumulator.keys()
