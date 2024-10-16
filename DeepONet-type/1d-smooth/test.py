
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter



errors = np.load('errors.npy')
N_fine = 257
grid_fine = np.linspace(0, 1, N_fine)


median_data = np.median(errors, axis=0)
p10 = np.percentile(errors, 10, axis=0)
p90 = np.percentile(errors, 90, axis=0)
pmin = np.min(errors, axis=0)
pmax = np.max(errors, axis=0)

x = grid_fine
plt.figure(figsize=(4, 3), dpi=150)
plt.plot(x, median_data, label='Median', color='#5A6CCA')
# plt.fill_between(x, pmin, pmax, color='#5A6CCA', alpha=0.3)
plt.fill_between(x, p10, p90, color='#5A6CCA', alpha=0.3)
plt.yscale("log")
plt.grid()
plt.tight_layout()
plt.show()
# plt.savefig('1d_smooth_errors')