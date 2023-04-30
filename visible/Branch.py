import matplotlib.pyplot as plt
import numpy as np


x = np.array([ 'branch(a)', 'branch(b)', 'branch(c)','branch(a+b+c)'])
bci = np.array([72.03, 73.30, 73.88,78.20])
bci_sem = [4.22,3.55,4.25,3.59]

hgd = np.array([93.52, 92.05, 96.34])
hgd_sem = [1.63,1.38,1.35,1.32,0.96]

# 绘制两组柱状图
fig, ax = plt.subplots(nrows=1,ncols=1)
num = 0
ax.bar(x,bci,yerr=bci_sem,align='center',width=0.5,error_kw={"ecolor": "black", "linewidth": 1},color = '#3b6291')
ax.errorbar(x, bci, yerr=bci_sem,color='black', ecolor = 'black',fmt='--o', linewidth=1, markersize=3, capsize=3,elinewidth=0.5,capthick=1)
ax.set_ylim([60, 85])
# 绘制 y 轴对应刻度的虚线
y = [60,65,70,75,80,85]

ax.set_yticks(y)
ax.set_yticklabels( y,fontfamily='Arial', fontsize=10)
for yi in y:
    ax.axhline(y=yi, linestyle='dotted', linewidth=1, color='gray', alpha=0.3)
ax.set_ylabel('Accuracy(%)', fontfamily='Arial', fontsize=10)
plt.xticks(rotation=0)
ax.set_xticklabels(x, fontfamily='Arial', fontsize=10)
ax.tick_params(axis='both', which='major', direction='out', labelsize=10, pad=5)
num=num+1

#在柱状图上加数据标签
text_strs = ["{:.2f}".format(val) for val in bci]
for i in range(len(x)):
    ax.text(i, bci[i] + 0.2, text_strs[i], ha='center',fontfamily='Arial', fontsize=8)


fig.subplots_adjust(wspace=0.5,hspace=0.3)


plt.show()

