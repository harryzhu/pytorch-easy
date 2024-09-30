#!/usr/bin/env python
# coding: utf-8

import os
import time
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

import config as CFG
from ImageByFolderDataset import *
from ImageByFolderNet import *
from ImageByFolderResNet import *

#model = ImageByFolderNet()
model = ImageByFolderResNet()

model_name = get_model_name(model)
model_output_dir = f'output/{model_name}'
print(model_output_dir)
if not os.path.isdir(model_output_dir):
	os.makedirs(model_output_dir)

if os.path.exists(f'{model_output_dir}/test_ok'):
	shutil.rmtree(f'{model_output_dir}/test_ok')
if os.path.exists(f'{model_output_dir}/test_error'):
	shutil.rmtree(f'{model_output_dir}/test_error')

test(model,all_test_images,model_output_dir)