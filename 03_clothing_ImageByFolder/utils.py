import os
import time
import glob
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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


def save_append(fpath,data,mode='a'):
	if not os.path.isdir(os.path.dirname(fpath)):
		os.mkdir(os.path.dirname(fpath))
	with open(fpath, mode) as f:
		f.write(data)
		f.close()

def get_latest_file(pattern):
	flist = glob.glob(pattern)
	fmtime = 0
	fname = ""
	for pth in flist:
		pth_mtime = os.path.getmtime(pth)
		if pth_mtime > fmtime:
			fmtime = pth_mtime
			fname = pth
	return fname

def get_model_name(model_instance):
	model_name = type(model_instance).__name__
	print(f'model_name: {model_name}')

	if model_name == "" or model_name is None:
		raise ValueError("ERROR: cannot get the model name.")
	return model_name

def plot_figure(param={'figsize':(30,20),'x':[],'y':[],'title':"",'xlabel':"",'ylabel':"",'savefig_path':""}):
	plt.figure(figsize=param['figsize'])
	plt_x = param['x']
	plt_y = param['x']
	plt.plot(plt_x, plt_y, marker='o')
	plt.title(param['title'])
	plt.xlabel(param['xlabel'])
	plt.ylabel(param['ylabel'])
	plt.xticks(size=22)
	plt.yticks(size=22)
	plt.grid(True)
	lr_dir = os.path.dirname(param['savefig_path'])
	if not os.path.isdir(lr_dir):
		os.makedirs(lr_dir)
	plt.savefig(param['savefig_path'])
	plt.close()



def train(net, loss_fn, train_iter, optimizer, epochs,output_dir):
	accumulator = Accumulator(['train_loss','vali_loss','train_acc','vali_acc'])
	device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
	print('train on device:', device)
	net = net.to(device)
	epoch_start = 0
	model_path = None
	latest_pth = get_latest_file(f'{output_dir}/*.pth')
	if latest_pth !="":
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
	fig_loss_reset_after_save = []
	batch_count = len(train_iter)
	print('batch_count:',len(train_iter))
	for epoch in range(epoch_start,epochs):
		time_train_start = time.time()
		len_train = 0
		len_vali = 0

		net.train()
		epoch_loss.clear()
		#accumulator['train_loss'].clear()
		#accumulator['train_acc'].clear()
		correct_num = 0
		
		
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

			fig_loss_reset_after_save.append(sum(epoch_loss)/len(epoch_loss))


			#print(f'{ iter_batch }: { iter_batch * CFG.batch_size }:')
			#print(':epoch_loss:',len(epoch_loss),epoch_loss)
			#print('correct_num:',correct_num)
			#print('len_train:',len_train)


		
		print('train_loss_size:',len(accumulator['train_loss']))
		plot_figure(param = {
			'figsize': (30,20),
			'x': range(epoch_start, epoch_start+len(accumulator['train_loss'])),
			'y': accumulator['train_loss'],
			'title': f'Loss Function Curve - Epoch: {epoch}',
			'xlabel': f'Batch: 0 - { len(accumulator["train_loss"]) }, Batch Size: {CFG.batch_size}',
			'ylabel': 'Loss',
			'savefig_path': f'{output_dir}/lr_{ optimizer.state_dict()["param_groups"][0]["lr"] }/1_loss_of_epoch_{epoch}.png'
			})

		
		

		print(f'----------- epoch: {epoch+1} start --------------')
		print(f'epoch: {epoch+1} / {epochs} train loss: { accumulator["train_loss"][-1] }')
		print(f'epoch: {epoch+1} / {epochs} train acc: { accumulator["train_acc"][-1] }')
		print(f'epoch: {epoch+1} / {epochs} train time: {int(time.time()-time_train_start)} sec')

			# save
		if (((epoch+1) % 10 == 0) or (epoch + 1 >= epochs)):
			torch.save(net.state_dict(), f'{output_dir}/model_{str(epoch+1)}.pth')
			#torch.save(net.state_dict(), './output/model_current.pth')

			print('fig_loss_reset_after_save:',len(fig_loss_reset_after_save))

			plot_figure(param = {
			'figsize': (30,20),
			'x': range(batch_count*epoch, batch_count*epoch + len(fig_loss_reset_after_save)),
			'y': fig_loss_reset_after_save,
			'title': f'Loss Function Curve - Reset After Save',
			'xlabel': f'Batch: {batch_count*epoch} - { batch_count*epoch + len(fig_loss_reset_after_save) }, Batch Size: {CFG.batch_size}',
			'ylabel': 'Loss',
			'savefig_path': f'{output_dir}/lr_{ optimizer.state_dict()["param_groups"][0]["lr"] }/2_loss_reset_after_save_{epoch+1}.png'
			})

			fig_loss_reset_after_save.clear()

	return accumulator


def test(model, all_test_images,output_dir):
	latest_pth = get_latest_file(f'{output_dir}/*.pth')
	model_path = None
	if latest_pth != "" and os.path.exists(latest_pth):
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
		if inference_image(image, model,output_dir):
			test_ok += 1.0
		else:
			test_error +=1.0


	ratio = 0.0
	if test_ok + test_error > 0:
		ratio = test_ok / (test_ok + test_error)
		print(f"[TEST] acc : {ratio:>.4f}, {test_ok} / { (test_ok + test_error) }")

	txt_1 = f'[TEST] acc: {ratio:>.4f} <= {test_ok} / { (test_ok + test_error) } <= {model_path}\n'
	save_append(f"{output_dir}/test_acc.txt", txt_1 , 'a+')

def inference_image(image_file,model,output_dir):
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
		infer_ok_path = f'{output_dir}/test_ok/{should_class}/{os.path.basename(image_file)}'
		if not os.path.isdir(os.path.dirname(infer_ok_path)):
			os.makedirs(os.path.dirname(infer_ok_path))
		shutil.copy(image_file, infer_ok_path)
		return True
	else:
		#print(f"ERROR: should {should_class}, but {all_classes[class_idx]}, {image_file}")
		infer_error_path =  f'{output_dir}/test_error/{should_class}/{os.path.basename(image_file)}'
		if not os.path.isdir(os.path.dirname(infer_error_path)):
			os.makedirs(os.path.dirname(infer_error_path))
		shutil.copy(image_file, infer_error_path)
		return False


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









