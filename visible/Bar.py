import matplotlib.pyplot as plt
import numpy as np

x= ['SSConv', 'HSConv', 'DMSConv']
bci = np.array([72.03, 73.46, 78.20])
bci_sem = [4.22, 4.30, 3.59]
hgd = np.array([93.52, 92.05, 96.34])
hgd_sem = [1.63, 2.05, 0.96]
data = [bci,hgd]
data_sem=[bci_sem,hgd_sem]
y_lim = [[60,85],[75,100]]
yx=[[60,65,70,75,80,85],[75,80,85,90,95,100]]

fig, axs = plt.subplots(nrows=1,ncols=2)
num = 0
for ax in axs.flat:
    dataset=data[num]

    ax.bar(x,dataset,yerr=data_sem[num],error_kw={"ecolor": "black", "linewidth": 1,"capsize": 3},align='center',width=0.5,color = '#3b6291')

    ax.set_ylim(y_lim[num])
    # 绘制 y 轴对应刻度的虚线
    y = yx[num]

    ax.set_yticks(y)
    ax.set_yticklabels( y,fontfamily='Arial', fontsize=10)
    for yi in y:
        ax.axhline(y=yi, linestyle='dotted', linewidth=1, color='gray', alpha=0.3)
    ax.set_ylabel('Accuracy(%)', fontfamily='Arial', fontsize=10)
    plt.xticks(rotation=0)

    ax.tick_params(axis='both', which='major', direction='out', labelsize=10, pad=5)
    text_strs = ["{:.2f}".format(val) for val in dataset]
    for i in range(len(x)):
        ax.text(i, dataset[i] + 0.2, text_strs[i], ha='center', fontfamily='Arial', fontsize=10)

    num=num+1
    fig.subplots_adjust(wspace=0.5,hspace=0.3)
    
plt.show()

