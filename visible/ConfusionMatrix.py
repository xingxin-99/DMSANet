# -*- coding: utf-8 -*-
"""
@Time    : 2021/11/18 0:33
@Author  : ONER
@FileName: plt_cm.py
@SoftWare: PyCharm
"""

# confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

fbcsp_bci_ho = np.array([[479, 99, 28, 42],
                [104, 456, 42, 46],
                [110, 59, 372, 107],
                [85, 40, 75, 448]])
deep_bci_ho = np.array([[505, 83, 39, 21],
                [84, 475, 58, 31],
                [63, 76, 453, 56],
                [65, 94, 78, 411]])
eeg_bci_ho = np.array([[447, 61, 85, 55],
                [47, 479, 77, 45],
                [52, 74, 454, 68],
                [51, 61, 66, 470]])
fbcnet_bci_ho= np.array([[501, 62, 42, 43],
                [70, 502, 40, 36],
                [52, 39, 475, 82],
                [69, 53, 51, 475]])
dmsa_bci_ho = np.array([[543, 58, 30, 17],
                [69, 511, 32, 36],
                [31, 37, 492, 88],
                [57, 61, 49, 481]])


cf = [fbcsp_bci_ho,deep_bci_ho,eeg_bci_ho,fbcnet_bci_ho,dmsa_bci_ho]
fig, axs = plt.subplots(nrows=1,ncols=5)
z=0
for ax in axs.flat:
    # bci :
    classes = ['left', 'right', 'feet', 'tongue']
    # hgd :
    # classes = ['left', 'right', 'feet', 'rest']
    confusion_matrix = cf[z]
    # 输入特征矩阵
    proportion = []
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)
    # print(np.sum(confusion_matrix[0]))
    # print(proportion)
    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(4, 4)  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(4, 4)
    # print(pshow)
    config = {
        "font.family": 'Arial',  # 设置字体类型
    }
    rcParams.update(config)
    im=ax.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    # plt.title('confusion_matrix')
    # ax.colorbar()
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes, fontsize=10)
    ax.set_yticks(tick_marks, classes, fontsize=10)

    thresh = confusion_matrix.max() / 2.
    iters = np.reshape([[[i, j] for j in range(4)] for i in range(4)], (confusion_matrix.size, 2))
    for i, j in iters:
        if (i == j):
            ax.text(j, i , format(confusion_matrix[i, j]), va='center', ha='center', fontsize=12, color='white',
                     weight=5)  # 显示对应的数字

        else:
            ax.text(j, i , format(confusion_matrix[i, j]), va='center', ha='center', fontsize=12)  # 显示对应的数字


    ax.set_ylabel('Actual classes', fontsize=10,fontfamily='Arial')
    ax.set_xlabel('Predict classes', fontsize=10,fontfamily='Arial')

    z=z+1
fig.subplots_adjust(wspace=0.2,hspace=0.2)

# 调整子图的位置和大小
# fig.subplots_adjust(right=0.85)

# # 在最右边的位置创建一个ax对象用于绘制colorbar
# cax = fig.add_axes([0.95, 0.15, 0.01, 0.5])

# # 绘制colorbar
# fig.colorbar(im, cax=cax)

plt.tight_layout()
plt.show()
