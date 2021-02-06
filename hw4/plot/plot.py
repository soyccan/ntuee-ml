import re
import numpy as np
import matplotlib.pyplot as plt

log = open('../work/model/word2vec-new.log').read()
T = [(x.group(1), x.group(2))
     for x in re.finditer(r'Train \| Loss:([0-9\.]+) Acc: ([0-9\.]+)', log)]
T = np.array(T, dtype='float')
V = [(x.group(1), x.group(2))
     for x in re.finditer(r'Valid \| Loss:([0-9\.]+) Acc: ([0-9\.]+)', log)]
V = np.array(V, dtype='float')

plt.plot(T[:, 0], label='train_loss')
plt.plot(V[:, 0], label='valid_loss')
plt.legend()
plt.show()

plt.plot(V[:, 1], label='valid_acc')
plt.plot(T[:, 1], label='train_acc')
plt.legend()
plt.show()
