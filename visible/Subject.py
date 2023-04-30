import matplotlib.pyplot as plt
import numpy as np
# 创建数据

# bci
x_bci = [1,2,3,4,5,6,7,8,9]

FBCSP_SVM_bci = [79.51,55.90,81.25,60.07,50.69,39.93,86.11,83.33,72.57]

deep_bci=[72.92 ,46.18 ,80.21 ,63.54 ,77.78 ,57.29 ,84.38 ,77.08 ,80.90 ]

eeg_bci=[77.78 ,61.11 ,91.32 ,68.06 ,70.83 ,56.94 ,72.22 ,74.65 ,69.44 ]

fbc_bci=[84.72 ,56.25 ,90.97 ,79.86 ,69.79 ,54.86 ,82.29 ,79.51 ,79.86 ]

DMSA_bci=[86.81 ,61.11 ,92.71 ,67.01 ,72.57 ,70.83 ,87.50 ,85.07 ,80.21 ]

# hgd
x_hgd = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
FBCSP_SVM_hgd = [86.88,68.75,92.50,88.13,79.38,93.75,76.10,79.38,79.38,81.88,71.25,87.50,74.84,57.5]

deep_hgd=[90.63 ,78.13 ,96.25 ,98.13 ,97.50 ,93.13 ,74.84 ,84.38 ,93.75 ,85.63 ,94.38 ,96.88 ,79.87 ,75.00
]

eeg_hgd=[75.00 ,85.63 ,97.50 ,99.38 ,100.00 ,94.38 ,76.10 ,93.13 ,93.75 ,91.88 ,84.38 ,91.25 ,88.05 ,80.00  ]

fbc_hgd=[88.13 ,84.38 ,98.75 ,98.75 ,95.00 ,95.00 ,84.91 ,97.50 ,86.25 ,93.75 ,79.38 ,95.00 ,88.05 ,81.88]

DMSA_hgd=[95.63 ,96.25 ,99.38 ,99.38 ,100.00 ,97.50 ,94.34 ,99.38 ,98.13 ,92.50 ,91.25 ,97.50 ,99.37 ,88.13  ]

FBCSP_SVM=[FBCSP_SVM_bci,FBCSP_SVM_hgd]
deep=[deep_bci,deep_hgd]
eeg=[eeg_bci,eeg_hgd]
fbc=[fbc_bci,fbc_hgd]
DMSA=[DMSA_bci,DMSA_hgd]
x=[x_bci,x_hgd]


# 创建画布和子图
fig, axs = plt.subplots(nrows=1,ncols=2)
i=0
width = 0.15

for ax in axs.flat:
    z = np.array(x[i])
    # 绘制线条
    ax.bar(z-2*width, FBCSP_SVM[i],width,label='FBCSP-SVM')
    ax.bar(z- 1*width, deep[i], width, label='Deep ConvNet')
    ax.bar(z, eeg[i], width, label='EEGNet')
    ax.bar(z+ 1*width, fbc[i], width, label='FBCNet')
    ax.bar(z+ 2*width, DMSA[i], width, label='DMSANet',color = [0.1,0.1,0.2])






    # 设置坐标轴标签和标题
    ax.set_xlabel('Subjects',fontfamily='Arial', fontsize=10,fontweight='bold')
    ax.set_ylabel('Accuracy(%)',fontfamily='Arial', fontsize=10,fontweight='bold')
    # ax.set_title('My Plot')

    # 隐藏上和右坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    # ax.spines['bottom'].set_color('red')
    # ax.spines['left'].set_color('red')

    # 设置坐标轴范围
    if i==0:
        ax.set_xlim([0, 10])
        ax.set_ylim([30, 105])
        y = [30,40,50, 60, 70, 80, 90, 100]
        ax.text(0.5, -0.15, "(a) BCI-IV-2a", horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes,fontdict={'family': 'arial', 'size': 10, 'weight': 'bold', 'color': 'black'})

    if i==1:
        ax.set_xlim([0, 15])
        ax.set_ylim([50, 105])
        y=[50,60,70,80,90,100]
        ax.text(0.5, -0.15, "(b) HGD", horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontdict={'family': 'arial', 'size': 10, 'weight': 'bold', 'color': 'black'})
    # 设置刻度

    # 绘制 y 轴对应刻度的虚线
    for yi in y:
        ax.axhline(y=yi, linestyle='dotted',linewidth=1, color='gray', alpha=0.3)

    ax.set_xticks(x[i])
    ax.set_yticks(y)
    ax.set_yticklabels(y,fontfamily='Arial', fontsize=10)
    ax.set_xticklabels(x[i],fontfamily='Arial', fontsize=10)
    ax.tick_params(axis='both', which='major',direction='in', labelsize=10,pad=5)
    ax.legend( loc='upper center', ncol=5,fontsize=10)


    i=i+1

# 添加图例

plt.savefig('C:\\Users\\BCIgroup\\Desktop\\star\\mingan.pdf',bbox_inches='tight')
# 显示图像
plt.show()

