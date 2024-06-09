import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec

# 混淆矩阵数据
confusion_matrices = {
    'FBCSP-SVM': np.array([[3208, 2192], [2043, 3357]]),
    'DeepConvNet': np.array([[3429, 1971], [1831, 3569]]),
    'EEGNet': np.array([[3633, 1767], [1713, 3687]]),
    'EEG-Inception': np.array([[3507, 1893], [1666, 3734]]),
    'FBCNet': np.array([[3662, 1738], [1632, 3768]]),
    'EEG Conformer': [
    [4011, 1389],
    [1614, 3786]
],
    'DMSANet': np.array([[3735	,1665],[1478,	3922]])
}
#
# [[3656, 1744], [1457, 3943]]
# 'DMSANet': np.array([[3791, 1609], [1416, 3984]]),
# confusion_matrices = {
#     'FBCSP-SVM': np.array([[479, 99, 28, 42],
#                 [104, 456, 42, 46],
#                 [110, 59, 372, 107],
#                 [85, 40, 75, 448]]),
#     'DeepConvNet': np.array([[505, 83, 39, 21],
#                 [84, 475, 58, 31],
#                 [63, 76, 453, 56],
#                 [65, 94, 78, 411]]),
#     'EEGNet': np.array([[447, 61, 85, 55],
#                 [47, 479, 77, 45],
#                 [52, 74, 454, 68],
#                 [51, 61, 66, 470]]),
#     'EEG-Inception': np.array([
#     [477, 99, 39, 33],
#     [88, 478, 56, 26],
#     [97, 92, 409, 50],
#     [80, 91, 85, 392]
# ]),
#     'FBCNet': np.array([[501, 62, 42, 43],
#                 [70, 502, 40, 36],
#                 [52, 39, 475, 82],
#                 [69, 53, 51, 475]]),
#     'EEG Conformer': [
#         [506, 65, 45, 32],
#         [86, 447, 61, 54],
#         [69, 64, 432, 83],
#         [72, 59, 46, 471]
#     ],
#
#     'DMSANet': np.array([[543, 58, 30, 17],
#                 [69, 511, 32, 36],
#                 [31, 37, 492, 88],
#                 [57, 61, 49, 481]])
# }

# confusion_matrices = {
#     'FBCSP-SVM': np.array([[479, 99, 28, 42],
#                 [104, 456, 42, 46],
#                 [110, 59, 372, 107],
#                 [85, 40, 75, 448]]),
#     'DeepConvNet': np.array([[505, 83, 39, 21],
#                 [84, 475, 58, 31],
#                 [63, 76, 453, 56],
#                 [65, 94, 78, 411]]),
#     'EEGNet': np.array([[447, 61, 85, 55],
#                 [47, 479, 77, 45],
#                 [52, 74, 454, 68],
#                 [51, 61, 66, 470]]),
#     'FBCNet': np.array([[501, 62, 42, 43],
#                 [70, 502, 40, 36],
#                 [52, 39, 475, 82],
#                 [69, 53, 51, 475]]),
#     'EEGInception': np.array([
#     [477, 99, 39, 33],
#     [88, 478, 56, 26],
#     [97, 92, 409, 50],
#     [80, 91, 85, 392]
# ]),
#     'DMSANet': np.array([[543, 58, 30, 17],
#                 [69, 511, 32, 36],
#                 [31, 37, 492, 88],
#                 [57, 61, 49, 481]])
# }


confusion_matrices = {
    'FBCSP-SVM': np.array([
    [452, 96, 9, 2],
    [76, 473, 4, 6],
    [37, 32, 440, 51],
    [19, 30, 52, 459]
]),
    'DeepConvNet': np.array([
    [498, 61, 0, 0],
    [52, 503, 4, 0],
    [16, 39, 476, 29],
    [14, 28, 15, 503]
]),
    'EEGNet': np.array([
    [483, 72, 1, 3],
    [47, 510, 1, 1],
    [10, 13, 495, 42],
    [10, 11, 28, 511]
] ),
    'EEG-Inception': np.array([
    [495, 61, 2, 1],
    [60, 497, 1, 1],
    [16, 24, 485, 35],
    [20, 9, 19, 512]
]  ),
    'FBCNet': np.array( [
    [482, 71, 1, 5],
    [45, 511, 0, 3],
    [6, 10, 494, 50],
    [4, 5, 13, 538]
]  ),

    'EEG Conformer': np.array([
    [525, 31, 2, 1],
    [28, 530, 1, 0],
    [9, 13, 508, 30],
    [10, 9, 21, 520]
] ),
    'DMSANet': np.array([
    [537, 21, 1, 0],
    [9, 547, 3, 0],
    [4, 10, 530, 16],
    [4, 9, 5, 542]
]  )
}
# 创建子图

num_matrices = len(confusion_matrices)
fig, axes = plt.subplots(1, 7, figsize=(num_matrices * 2,2))
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1)

# 绘制每个混淆矩阵
for i, (model, matrix) in enumerate(confusion_matrices.items()):
    ax = axes[i]
    proportion = []
    for i in matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)
    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)

    proportion = np.array(proportion).reshape(4, 4)  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(4, 4)


    im = ax.imshow(proportion, cmap='Blues', interpolation='nearest',norm=norm)
    # font_path = 'E:\\Star\\SimSun.ttc'
    # font_prop = FontProperties(fname=font_path)
    # 设置中文和英文的字体

    font_path_english = 'E:\\Star\\TimesNewRoman.ttf'
    font_prop_english = FontProperties(fname=font_path_english)
    # 添加标签
    ax.set_title(model,fontproperties='Arial',fontsize=8)
    # ax.set_xlabel('Predicted classes',fontproperties='Arial',fontsize=10)
    # ax.set_ylabel('Actual classes',fontproperties='Arial',fontsize=10)


    # 显示刻度标签
    # ax.set_xticks([0, 1])
    # ax.set_yticks([0, 1])
    # ax.set_xticklabels(['left', 'right'],fontproperties=font_prop_english,fontsize=10)
    # ax.set_yticklabels(['left', 'right'],fontproperties=font_prop_english,fontsize=10)

    ax.set_xticks([0, 1,2,3])
    ax.set_yticks([0, 1,2,3])
    ax.set_xticklabels(['left', 'right','feet','rest'],fontproperties='Arial',fontsize=8)
    ax.set_yticklabels(['left', 'right','feet','rest'],fontproperties='Arial',fontsize=8,rotation='vertical')



    # 显示数字
    iters = np.reshape([[[i, j] for j in range(4)] for i in range(4)], (pshow.size, 2))
    for i, j in iters:
        if (i == j):
            ax.text(j, i, format(pshow[i, j]), va='center', ha='center', fontproperties='Arial',fontsize=8, color='white',
                    )  # 显示对应的数字

        else:
            ax.text(j, i, format(pshow[i, j]), va='center', ha='center', fontproperties='Arial',fontsize=8)  # 显示对应的数字


# # 前面三个子图的总宽度 为 全部宽度的 0.9；剩下的0.1用来放置colorbar
# fig.subplots_adjust(right=0.90)
# #
# # #colorbar 左 下 宽 高
# l = 0.90
# b = 0.12
# w = 0.01
# h = 1 - 2*b
# #
# # # 对应 l,b,w,h；设置colorbar位置；
# rect = [l, b, w, h]
# cbar_ax = fig.add_axes(rect)
# cb = fig.colorbar(im,cbar_ax)
#
#
# # 设置colorbar标签字体等
# cb.ax.tick_params(labelsize=8)  # 设置色标刻度字体大小。
# font = {'family': 'Arial',
#         #       'color'  : 'darkred',
#         'color': 'black',
#         'weight': 'normal',
#         'size': 8,
#         }


plt.tight_layout()

# 显示图形
plt.show()
