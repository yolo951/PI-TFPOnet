
import numpy as np
import matplotlib.pyplot as plt

tfponet_loss = np.load(r'DeepONet-type\2d-smooth\saved_data\rel_l2_history.npy')
supervised_deeponet_loss = np.load(r'DeepONet-type\2d-smooth\saved_data\supervised_deeponet_rel_l2_history.npy')
unsupervised_deeponet_loss = np.load(r'DeepONet-type\2d-smooth\saved_data\unsupervised_deeponet_rel_l2_history.npy')
supervised_ionet_loss = np.load(r'DeepONet-type\2d-smooth\saved_data\supervised_ionet_rel_l2_history.npy')
unsupervised_ionet_loss = np.load(r'DeepONet-type\2d-smooth\saved_data\unsupervised_ionet_rel_l2_history.npy')

plt.plot(np.arange(0, len(tfponet_loss)*100, 100), tfponet_loss, linewidth=1.5, label='TFPONet(ours)', color='black', linestyle='-')
plt.plot(np.arange(0, len(supervised_deeponet_loss)*100, 100), supervised_deeponet_loss, linewidth=1.5, label='supervised DeepONet', color='red', linestyle='-')
plt.plot(np.arange(0, len(unsupervised_deeponet_loss)*100, 100), unsupervised_deeponet_loss, linewidth=1.5, label='unsupervised DeepONet', color='red', linestyle='--')
plt.plot(np.arange(0, len(supervised_ionet_loss)*100, 100), supervised_ionet_loss, linewidth=1.5, label='supervised IONet', color='blue', linestyle='-')
plt.plot(np.arange(0, len(unsupervised_ionet_loss)*100, 100), unsupervised_ionet_loss, linewidth=1.5, label='unsupervised IONet', color='blue', linestyle='--')
plt.legend(frameon=False)
plt.grid(True, alpha=0.3)
plt.xlabel('epochs', fontsize=12)
plt.ylabel('relative L2 norm', fontsize=12)
plt.savefig(r'DeepONet-type\2d-smooth\saved_data\all_rel_l2_history.png')
plt.yscale('log')
plt.show()