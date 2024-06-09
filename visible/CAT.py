import os

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pytorch_grad_cam import GradCAM
from scipy.io import loadmat, savemat
from matplotlib import mlab as mlab
import math

import torch
import torch.nn as nn
from torch.nn import Parameter
from torchsummary import summary

import sys
from torchsummary import summary
# from preprocess import import_data

from networks import DMSANet

font_path = 'E:\\Star\\Times New Roman.ttf'
font = FontProperties(fname=font_path)
plt.rcParams['font.size'] = 10


model = DMSANet()
sub =7
bestpath = os.path.join("E:\\Star\\star\\Coding\\DMSANet\\DMSANet_Best", "model_" + str(sub)+"_all.pth")
print(bestpath)
model.load_state_dict(torch.load(bestpath))
SPConv = dict(model.named_children())['SP'][0]
w = SPConv.weight.detach().view(120,22)#.cpu().numpy() #获取该卷积层对应的卷积核权重

w = torch.abs(w).mean(0)
w = (w-torch.mean(w,dim = 0,keepdim=True))/torch.std(w,dim=0,keepdim=True)

w = (w-torch.min(w))/(torch.max(w)-torch.min(w))




biosemi_montage = mne.channels.make_standard_montage('biosemi64')
index = [37, 9, 10, 46, 45, 44, 13, 12, 11, 47, 48, 49, 50, 17, 18, 31, 55, 54, 19, 30, 56, 29]
biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
biosemi_montage.dig = [biosemi_montage.dig[i+3] for i in index]
# picked_channels = [biosemi_montage.ch_names[i] for i in index]
info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250, ch_types='eeg')
# mne.viz.plot_montage(biosemi_montage,scale_factor=0.5)

mean_se = w.reshape(-1, 1)
evoked = mne.EvokedArray(mean_se, info)
evoked.set_montage(biosemi_montage)



im,cn = mne.viz.plot_topomap(w, evoked.info,  size = 3,show=False,contours = 2,cmap='RdBu_r',names=biosemi_montage.ch_names)
# plt.colorbar(im)
plt.show()
directory = "E:\\Star\\figure\\BREANet"

plt.savefig(os.path.join(directory, "all_sub" + str(sub)+ "_1.png"),dpi = 600,bbox_inches = 'tight')
# plt.savefig(os.path.join(directory, "all_sub" +str(sub)+ "_.pdf"),dpi = 600,bbox_inches = 'tight')


