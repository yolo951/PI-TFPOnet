
import numpy as np
import matplotlib.pyplot as plt

tfponet_loss = np.load('DeepONet-type/2d-singular/saved_data/rel_l2_history.npy')
supervised_deeponet_loss = np.load('DeepONet-type/2d-singular/saved_data/supervised_deeponet_rel_l2_history.npy')
unsupervised_deeponet_loss = np.load('DeepONet-type/2d-singular/saved_data/unsupervised_deeponet_rel_l2_history.npy')
supervised_ionet_loss = np.load('DeepONet-type/2d-singular/saved_data/supervised_ionet_rel_l2_history.npy')
unsupervised_ionet_loss = np.load('DeepONet-type/2d-singular/saved_data/unsupervised_ionet_rel_l2_history.npy')
supervised_fno_loss = np.load('DeepONet-type/2d-singular/saved_data/supervised_fno_rel_l2_history.npy')
unsupervised_fno_loss = np.load('DeepONet-type/2d-singular/saved_data/unsupervised_fno_rel_l2_history.npy')

plt.plot(np.arange(0, len(tfponet_loss)*100, 100), tfponet_loss, linewidth=2.5, label='PI-TFPONet(ours)', color='black', linestyle='-')
plt.plot(np.arange(0, len(supervised_deeponet_loss)*100, 100), supervised_deeponet_loss, linewidth=2.5, label='DeepONet', color='#427AB2', linestyle='-')
plt.plot(np.arange(0, len(unsupervised_deeponet_loss)*100, 100), unsupervised_deeponet_loss, linewidth=2.5, label='PI-DeepONet', color='#427AB2', linestyle='--')
plt.plot(np.arange(0, len(supervised_ionet_loss)*100, 100), supervised_ionet_loss, linewidth=2.5, label='IONet', color='#E56F5E', linestyle='-')
plt.plot(np.arange(0, len(unsupervised_ionet_loss)*100, 100), unsupervised_ionet_loss, linewidth=2.5, label='PI-IONet', color='#E56F5E', linestyle='--')
plt.plot(np.arange(0, len(supervised_fno_loss)*100, 100), supervised_fno_loss, linewidth=2.5, label='FNO', color='#F6C957', linestyle='-')
plt.plot(np.arange(0, len(unsupervised_fno_loss)*100, 100), unsupervised_fno_loss, linewidth=2.5, label='PI-FNO', color='#F6C957', linestyle='--')
plt.legend(frameon=False, bbox_to_anchor=(0.4, 0.7), ncol=2)
plt.grid(True, alpha=0.3)
plt.xlabel('epochs', fontsize=12)
plt.ylabel('relative L2 norm', fontsize=12)
plt.yscale('log')
plt.savefig('DeepONet-type/2d-singular/saved_data/2d-singular_all_rel_l2_history.png')
plt.show()

dt_tfpnet = 0.17
dt_deeponet = 0.04
dt_pideeponet = 0.36
dt_ionet = 0.07
dt_piionet = 0.38
dt_fno = 0.45
dt_pifno = 0.35
T = 800

fig = plt.figure()
plt.scatter(np.arange(0, T+1, dt_tfpnet*100), tfponet_loss[:int(T/(dt_tfpnet*100)+1)], label='PI-TFPONet(ours)', color='black')
plt.scatter(np.arange(0, T+1, dt_pideeponet*100), unsupervised_deeponet_loss[:int(T/(dt_pideeponet*100)+1)], linewidth=2.5, label='PI-DeepONet', color='#427AB2', linestyle='--')
plt.scatter(np.arange(0, T+1, dt_piionet*100), unsupervised_ionet_loss[:int(T/(dt_piionet*100)+1)], linewidth=2.5, label='PI-IONet', color='#E56F5E', linestyle='--')
plt.scatter(np.arange(0, T+1, dt_pifno*100), unsupervised_fno_loss[:int(T/(dt_pifno*100)+1)], linewidth=2.5, label='PI-FNO', color='#F6C957', linestyle='--')
plt.legend(bbox_to_anchor=(0.6, 0.7))
plt.xlabel('runtime(s)', fontsize=12)
plt.ylabel('relative L2 norm', fontsize=12)
plt.xscale('log')
ax = plt.gca()
ax.set_facecolor('#F2F2F2')
plt.savefig('DeepONet-type/2d-singular/saved_data/2d-singular_all_rel_l2_time.png')
plt.show()