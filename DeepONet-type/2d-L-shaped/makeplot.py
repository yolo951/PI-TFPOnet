import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np


def plot_domain_with_L_hole_no_line():
    fig, ax = plt.subplots()

    outer_square = patches.Rectangle((0, 0), 1, 1, edgecolor='black', facecolor='lightgray')
    ax.add_patch(outer_square)

    l_hole_coords = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.5), (0.5, 0.5), (0.5, 0.75), (0.25, 0.75)]
    l_hole = patches.Polygon(l_hole_coords, edgecolor='black', facecolor='white')
    ax.add_patch(l_hole)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    ax.text(0.61, 0.61, r'$\Omega_1$', fontsize=15, ha='center', va='center', color='black')
    ax.text(0.35, 0.35, r'$\Omega_2$', fontsize=15, ha='center', va='center', color='black')

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    plt.savefig('DeepONet-type/2d-L-shaped/saved_data/2d_Lshaped.png')
    plt.show()

# plot the domain with L-shaped hole
# plot_domain_with_L_hole_no_line()

tfponet_loss = np.load('DeepONet-type/2d-L-shaped/saved_data/rel_l2_history.npy')
supervised_deeponet_loss = np.load('DeepONet-type/2d-L-shaped/saved_data/supervised_deeponet_rel_l2_history.npy')
unsupervised_deeponet_loss = np.load('DeepONet-type/2d-L-shaped/saved_data/unsupervised_deeponet_rel_l2_history.npy')
supervised_ionet_loss = np.load('DeepONet-type/2d-L-shaped/saved_data/supervised_ionet_rel_l2_history.npy')
unsupervised_ionet_loss = np.load('DeepONet-type/2d-L-shaped/saved_data/unsupervised_ionet_rel_l2_history.npy')
supervised_fno_loss = np.load('DeepONet-type/2d-L-shaped/saved_data/supervised_fno_rel_l2_history.npy')
unsupervised_fno_loss = np.load('DeepONet-type/2d-L-shaped/saved_data/unsupervised_fno_rel_l2_history.npy')

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
plt.savefig('DeepONet-type/2d-L-shaped/saved_data/2d-L-shaped_all_rel_l2_history.png')
plt.show()

dt_tfpnet = 0.25
dt_deeponet = 0.05
dt_pideeponet = 0.36
dt_ionet = 0.06
dt_piionet = 0.37
dt_fno = 0.45
dt_pifno = 1.38
T = 2000

fig = plt.figure()
plt.scatter(np.arange(0, T+1, dt_tfpnet*100), tfponet_loss[:int(T/(dt_tfpnet*100)+1)], label='PI-TFPONet(ours)', color='black')
plt.scatter(np.arange(0, T+1, dt_pideeponet*100), unsupervised_deeponet_loss[:int(T/(dt_pideeponet*100)+1)], linewidth=2.5, label='PI-DeepONet', color='#427AB2', linestyle='--')
plt.scatter(np.arange(0, T+1, dt_piionet*100), unsupervised_ionet_loss[:int(T/(dt_piionet*100)+1)], linewidth=2.5, label='PI-IONet', color='#E56F5E', linestyle='--')
plt.scatter(np.arange(0, T+1, dt_pifno*100), unsupervised_fno_loss[:int(T/(dt_pifno*100)+1)], linewidth=2.5, label='PI-FNO', color='#F6C957', linestyle='--')
plt.legend(bbox_to_anchor=(0.6, 0.7))
plt.grid(True, alpha=0.3)
plt.xlabel('runtime(s)', fontsize=12)
plt.ylabel('relative L2 norm', fontsize=12)
plt.xscale('log')
plt.yscale('log')
ax = plt.gca()
ax.set_facecolor('#F2F2F2')
plt.savefig('DeepONet-type/2d-L-shaped/saved_data/2d-L-shaped_all_rel_l2_time.png')
plt.show()