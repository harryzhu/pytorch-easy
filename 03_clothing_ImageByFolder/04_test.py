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

model = ImageByFolderNet()

if os.path.exists("output/test_ok"):
	shutil.rmtree("output/test_ok")
if os.path.exists("output/test_error"):
	shutil.rmtree("output/test_error")

test(model,all_test_images)