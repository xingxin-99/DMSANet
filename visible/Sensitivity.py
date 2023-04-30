import matplotlib.pyplot as plt
import numpy as np
# 创建数据
x = [20, 40, 60, 80, 100]
FBCSP_SVM = [65.50 ,	72.70 ,	79.26	,78.91	,81.50 ]
FBCSP_SVM_std=np.array([ 3.98,3.61, 2.74, 3.22	, 2.96])

deep=[64.06,	76.89,	81.44,	84.98,	88.46]
deep_std= np.array([3.66, 3.62, 3.42, 2.93	, 2.31])
eeg=[61.71 ,	75.43 ,	82.35 ,	85.57 ,	89.31 ]
eeg_std=np.array([4.19	,3.60	,2.06	,2.62	,2.17])
fbc=[74.30 ,	83.06 ,	85.12 ,	88.65 ,	90.479]
fbc_std=np.array([3.11	,2.50	,2.41	,2.22	,1.74,])
DMSA=[79.89,	87.00,	91.55,	93.38,	96.34]
DMSA_std=np.array([3.04,2.49,1.93,1.48	,0.96])

# 创建画布和子图
fig, ax = plt.subplots(figsize=(3.5, 3.5))

# 绘制线条
ax.errorbar(x, FBCSP_SVM,yerr=FBCSP_SVM_std,fmt='--o',linewidth=1, markersize=3, capsize=3,elinewidth=1,capthick=1, label='FBCSP-SVM')
ax.errorbar(x, deep,yerr=deep_std, fmt='--o',linewidth=1, markersize=3, capsize=3,elinewidth=1,capthick=1,label='Deep ConvNet')
ax.errorbar(x, eeg,yerr=eeg_std,fmt='--o', linewidth=1, markersize=3, capsize=3,elinewidth=1,capthick=1,label='EEGNet')
ax.errorbar(x, fbc,yerr=fbc_std, fmt='--o', linewidth=1,markersize=3, capsize=3,elinewidth=1,capthick=1,label='FBCNet')
ax.errorbar(x, DMSA,yerr=DMSA_std,fmt='-o',linewidth=1, markersize=3, capsize=3,elinewidth=1,capthick=1, color='black',label='DMSANet')

# 设置坐标轴标签和标题
ax.set_xlabel('Training Data(%)',fontfamily='Arial', fontsize=10,fontweight='bold')
ax.set_ylabel('Accuracy(%)',fontfamily='Arial', fontsize=10,fontweight='bold')
# ax.set_title('My Plot')

# 隐藏上和右坐标轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)
# ax.spines['bottom'].set_color('red')
# ax.spines['left'].set_color('red')


# plt.rcParams['text.color'] = 'darkgray'  # 设置字体颜色
# plt.rcParams['axes.labelcolor'] = 'darkgray'  # 设置标签颜色
# plt.rcParams['xtick.color'] = 'darkgray'  # 设置x轴标签颜色
# plt.rcParams['ytick.color'] = 'darkgray'  # 设置y轴标签颜色
# 设置坐标轴范围
ax.set_xlim([15, 105])
ax.set_ylim([55, 100])
y=[60,70,80,90,100]
# 设置刻度

# 绘制 y 轴对应刻度的虚线
for yi in y:
    ax.axhline(y=yi, linestyle='dotted',linewidth=1, color='gray', alpha=0.3)

ax.set_xticks(x)
ax.set_yticks(y)
ax.set_yticklabels(y,fontfamily='Arial', fontsize=10)
ax.set_xticklabels(x,fontfamily='Arial', fontsize=10)
ax.tick_params(axis='both', which='major',direction='in', labelsize=10,pad=5)

# 添加图例
ax.legend(loc='lower right', fontsize=10 )
plt.savefig('C:\\Users\\BCIgroup\\Desktop\\star\\mingan.pdf',bbox_inches='tight')
# 显示图像
plt.show()

