from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection
import numpy as np
import warnings
warnings.filterwarnings("ignore")

dmsanet_bci_acc = [0.868055556, 0.611111111, 0.927083333, 0.670138889, 0.725694444, 0.708333333, 0.875, 0.850694444, 0.802083333]
fbcsp_bci_acc = [0.795138889, 0.559027778, 0.8125, 0.600694, 0.5069444, 0.399305555, 0.86111, 0.833333, 0.72569]
eeg_bci_acc = [0.777777778, 0.611111111, 0.913194444, 0.680555556, 0.708333333, 0.569444444, 0.722222222, 0.746527778, 0.694444444]
deep_bci_acc =[0.729166667, 0.461805556, 0.802083333, 0.635416667, 0.777777778, 0.572916667, 0.84375, 0.770833333, 0.809027778]
fbc_bci_acc = [0.847222222, 0.5625, 0.909722222, 0.798611111, 0.697916667, 0.548611111, 0.822916667, 0.795138889, 0.798611111]

# 计算与dmsanet的p值
p_values = []
for acc in [fbcsp_bci_acc, eeg_bci_acc, deep_bci_acc, fbc_bci_acc]:
    _, p_value = wilcoxon(acc, dmsanet_bci_acc)
    p_values.append(p_value)

# 进行FDR检验
rejected, corrected_p_values = fdrcorrection(p_values)

print('p-values:', p_values)
print('corrected p-values:', corrected_p_values)
print('rejected:', rejected)
