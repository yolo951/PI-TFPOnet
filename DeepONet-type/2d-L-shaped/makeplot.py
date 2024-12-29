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

plot_domain_with_L_hole_no_line()

tfponet_loss = np.load(r'DeepONet-type\2d-L-shaped\saved_data\rel_l2_history.npy')
supervised_deeponet_loss = np.load(r'DeepONet-type\2d-L-shaped\saved_data\supervised_deeponet_rel_l2_history.npy')
unsupervised_deeponet_loss = np.load(r'DeepONet-type\2d-L-shaped\saved_data\unsupervised_deeponet_rel_l2_history.npy')
supervised_ionet_loss = np.load(r'DeepONet-type\2d-L-shaped\saved_data\supervised_ionet_rel_l2_history.npy')
unsupervised_ionet_loss = np.load(r'DeepONet-type\2d-L-shaped\saved_data\unsupervised_ionet_rel_l2_history.npy')

plt.plot(np.arange(0, len(tfponet_loss)*100, 100), tfponet_loss, linewidth=1.5, label='TFPONet(ours)', color='black', linestyle='-')
plt.plot(np.arange(0, len(supervised_deeponet_loss)*100, 100), supervised_deeponet_loss, linewidth=1.5, label='supervised DeepONet', color='red', linestyle='-')
plt.plot(np.arange(0, len(unsupervised_deeponet_loss)*100, 100), unsupervised_deeponet_loss, linewidth=1.5, label='unsupervised DeepONet', color='red', linestyle='--')
plt.plot(np.arange(0, len(supervised_ionet_loss)*100, 100), supervised_ionet_loss, linewidth=1.5, label='supervised IONet', color='blue', linestyle='-')
plt.plot(np.arange(0, len(unsupervised_ionet_loss)*100, 100), unsupervised_ionet_loss, linewidth=1.5, label='unsupervised IONet', color='blue', linestyle='--')
plt.legend(frameon=False, bbox_to_anchor=(0.75, 0.5))
plt.grid(True, alpha=0.3)
plt.xlabel('epochs', fontsize=12)
plt.ylabel('relative L2 norm', fontsize=12)
plt.savefig(r'DeepONet-type\2d-L-shaped\saved_data\2d-L-shaped_all_rel_l2_history.png')
plt.yscale('log')
plt.show()
