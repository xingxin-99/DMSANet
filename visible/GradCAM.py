# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:17:29 2019

@author: JIAN
"""
import os
import torch
import scipy.io as sio
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import pickle
from pytorch_grad_cam import GradCAM

from networks import DMASNet

#更改图片中字体的格式与大小
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 9
plt.rcParams.update({'font.size': 9, 'font.family': 'Arial'})

plt.rcParams.update({'font.size': 12})

torch.cuda.empty_cache()
torch.manual_seed(0)



# 导入模型并加载最好的模型
model =DMSANet(nChan=22, nTime=1000, nClass=4)
model.load_state_dict(torch.load('C:/Users/BCIgroup/Desktop/star/Coding/FBCNet/FBCNet-master/codes/netInitModels/DMSANet_0.pth'))
target_layer = [model.FC[0]]

# 载入数据
dataPath = 'C:/Users/BCIgroup/Desktop/star/Coding/FBCNet/FBCNet-master/data/bci42a/rawPython/00000.dat'
channelnum = 22
samplelength = 1000


# # 处理数据

with open(dataPath, 'rb') as fp:
    d = pickle.load(fp)
data = d['data']
rawdata = d['data']
label = d['label']
print(label)
data = torch.from_numpy(data)
# 插入维度
data = data.unsqueeze(0)
# 【1,3,256,256】
data = data.unsqueeze(0)
# 计算CAM
cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)
target_category = label
grayscale_cam = cam(input_tensor=data)[0]
fig, axs = plt.subplots(nrows=1 , ncols=1)

# for ax in axs.flat:
    # 设置X轴和Y轴的标签
axs.set_ylabel('Electrode',fontfamily='Arial', fontsize=8)
axs.set_xlabel('Time(s)',fontfamily='Arial', fontsize=8)
#设置y轴坐标间隔
thespan = np.percentile(rawdata, 98)
# thespan=100
yttics = np.zeros(channelnum)
for i in range(channelnum):
    yttics[i] = i * thespan

axs.set_ylim([-thespan, thespan * channelnum])  # 设置坐标轴范围
axs.set_xlim([0, 1001])
labels = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
          'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
          'CP3', 'CP1', 'CPz','CP2', 'CP4', 'P1',
          'Pz', 'P2', 'POz']
axs.set_yticks(yttics,rotation=45)
axs.set_yticklabels(labels,fontfamily='Arial', fontsize=8)
axs.set_xticks([index - 0.5 for index in [0,250,500,750,1000]])
axs.set_xticklabels([0, 1, 2, 3, 4],fontfamily='Arial', fontsize=8)
# ax.tick_params(axis='y', labelsize=8)
xx = np.arange(1, samplelength + 1)


for i in range(0, channelnum):
    y = rawdata[i, :] + thespan * (i)
    dydx = grayscale_cam[i, :]
    # im = ax.imshow(dydx, cmap='jet')

    points = np.array([xx, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap='viridis', norm=norm)

    lc.set_array(dydx)
    lc.set_linewidth(0.5)
    axs.add_collection(lc)

fig.subplots_adjust(wspace=0.5,hspace=0.3)

cbar = fig.colorbar( lc,ax=axs)
cbar.ax.tick_params(labelsize=8)
plt.rcParams['font.family'] = 'Arial'
cbar.set_label('', fontsize=8, fontfamily='Arial')
plt.savefig('C:/Users/BCIgroup/Desktop/star/output.pdf', dpi=300, bbox_inches='tight')

plt.show()
























