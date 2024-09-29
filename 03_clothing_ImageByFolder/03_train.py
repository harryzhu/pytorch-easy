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
num_epoch = 300
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

accumulator = train(model, loss_fn, dataloader_train, optimizer, num_epoch)
print("------------------")
#print(accumulator)

