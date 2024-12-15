
import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
from scipy import interpolate
from scipy import sparse
from scipy.sparse import linalg
from math import sqrt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_points(N):
    x0 = np.linspace(0, 1, 1+4*N)
    x0 = x0[1::2]
    x_b = np.hstack([x0, np.zeros(2*N), x0, np.ones(2*N)])
    y_b = np.hstack([np.zeros(2*N), x0, np.ones(2*N), x0])
    return np.column_stack([x_b, y_b]), np.column_stack([0.5*np.ones(2*N), x0])

def create_interpolator(f, grid):
    interpolators = [
        interpolate.RegularGridInterpolator((grid, grid), f[i], bounds_error=False, fill_value=None)
        for i in range(f.shape[0])
    ]

    def F(x, y):
        points = np.column_stack((x, y))  # shape = (num_points, 2)
        interpolated_values = np.array([interp(points) for interp in interpolators])
        return interpolated_values  # shape = (batch_size, num_points)
    return F

def generate_coeff_b(F, points):
    N = f.shape[-1]-1
    h = 1/N
    # generate index to gather from C
    idx1 = np.arange(4*N)
    base = np.arange(N)*4*N
    offsets = np.arange(4)
    idx2 = (base[:, None] + offsets).flatten()
    idx3 = np.arange((N-1)*4*N, 4*N**2)
    base = np.arange(1, N+1)*4*N
    offsets = np.array([-4, -3, -2, -1])
    idx4 = (base[:, None] + offsets).flatten()
    idx = np.hstack((idx1, idx2, idx3, idx4))
    idx = np.repeat(idx.reshape((-1, 4)), 2, axis=0).flatten()

    # generate coeff
    centor_x = np.where(points[:, 0]==0, 1/2/N, np.where(points[:, 0]==1, 1-h/2, points[:, 0]//h*h+h/2))
    centor_y = np.where(points[:, 1]==0, 1/2/N, np.where(points[:, 1]==1, 1-h/2, points[:, 1]//h*h+h/2))
    f0 = F(centor_x, centor_y)
    c0 = np.where(centor_x<1/2, 16*1000.0, 1000.0)
    mu0 = np.sqrt(c0)
    coeff = np.column_stack((np.exp(mu0*(points[:, 0]-centor_x)-mu0*h/2),
                             np.exp(-mu0*(points[:, 0]-centor_x)-mu0*h/2),
                             np.exp(mu0*(points[:, 1]-centor_y)-mu0*h/2),
                             np.exp(-mu0*(points[:, 1]-centor_y)-mu0*h/2)))
    b0 = np.where(points[:, 0]>=1/2, 2*(1-points[:, 0]), 0.0)
    rhs = (-f0/c0 + b0)*np.exp(-mu0*h/2)
    return idx, coeff, rhs


def generate_coeff_i(F, points, alpha, beta):
    N = f.shape[-1]-1
    h = 1/N
    # generate index to gather from C
    base = np.arange(2*N-4, 4*N**2-2*N-3, 4*N)
    offsets = np.arange(4)
    idx1 = (base[:, None] + offsets).flatten()
    base = np.arange(2*N, 4*N**2-2*N+1, 4*N)
    offsets = np.arange(4)
    idx2 = (base[:, None] + offsets).flatten()
    idx = np.hstack((idx1, idx2))
    idx = np.repeat(idx.reshape((-1, 8)), 2, axis=0).flatten()
    idx = np.hstack((idx, idx))

    # generate coeff
    centor_x_l = (0.5-h/2)*np.ones(2*N)
    centor_x_r = (0.5+h/2)*np.ones(2*N)
    centor_y_l = points[:, 1]//h*h+h/2
    centor_y_r = points[:, 1]//h*h+h/2
    f0_l = F(centor_x_l, centor_y_l)
    c0_l = np.where(centor_x_l<1/2, 16*1000.0, 1000.0)
    mu0_l = np.sqrt(c0_l)
    f0_r = F(centor_x_r, centor_y_r)
    c0_r = np.where(centor_x_r<1/2, 16*1000.0, 1000.0)
    mu0_r = np.sqrt(c0_r)
    mu = np.maximum(mu0_l, mu0_r)
    # jump of u
    coeff_l = np.column_stack((np.exp(mu0_l*(points[:, 0]-centor_x_l)-mu*h/2),
                               np.exp(-mu0_l*(points[:, 0]-centor_x_l)-mu*h/2),
                               np.exp(mu0_l*(points[:, 1]-centor_y_l)-mu*h/2),
                               np.exp(-mu0_l*(points[:, 1]-centor_y_l)-mu*h/2)))
    coeff_r = np.column_stack((np.exp(mu0_r*(points[:, 0]-centor_x_r)-mu*h/2),
                               np.exp(-mu0_r*(points[:, 0]-centor_x_r)-mu*h/2),
                               np.exp(mu0_r*(points[:, 1]-centor_y_r)-mu*h/2),
                               np.exp(-mu0_r*(points[:, 1]-centor_y_r)-mu*h/2)))
    coeff = np.concatenate((-coeff_l, coeff_r), axis=-1)
    rhs_l = -f0_l/c0_l*np.exp(-mu*h/2)
    rhs_r = -f0_r/c0_r*np.exp(-mu*h/2)
    rhs = rhs_r - rhs_l + alpha*np.exp(-mu*h/2)
    # jump of du
    coeff_l = np.column_stack((mu0_l*np.exp(mu0_l*(points[:, 0]-centor_x_l)-mu*h/2),
                               -mu0_l*np.exp(-mu0_l*(points[:, 0]-centor_x_l)-mu*h/2),
                               mu0_l*np.exp(mu0_l*(points[:, 1]-centor_y_l)-mu*h/2),
                               -mu0_l*np.exp(-mu0_l*(points[:, 1]-centor_y_l)-mu*h/2)))
    coeff_r = np.column_stack((mu0_r*np.exp(mu0_r*(points[:, 0]-centor_x_r)-mu*h/2),
                               -mu0_r*np.exp(-mu0_r*(points[:, 0]-centor_x_r)-mu*h/2),
                               mu0_r*np.exp(mu0_r*(points[:, 1]-centor_y_r)-mu*h/2),
                               -mu0_r*np.exp(-mu0_r*(points[:, 1]-centor_y_r)-mu*h/2)))
    coeff = np.concatenate((coeff, np.concatenate((-coeff_l, coeff_r), axis=-1)), axis=0)
    rhs = np.concatenate((rhs, np.tile(beta*np.exp(-mu*h/2), (rhs.shape[0], 1))), axis=-1)
    return idx, coeff, rhs

if __name__ == '__main__':
    N = 16
    points_b, points_i = generate_points(N)
    f = np.load(r'DeepONet-type\2d-singular\f.npy')
    F = create_interpolator(f, np.linspace(0, 1, N+1))
    idx_b, coeff_b, rhs_b = generate_coeff_b(F, points_b)
    idx_i, coeff_i, rhs_i = generate_coeff_i(F, points_i, alpha=1.0, beta=0.0)
    # np.savez(r"DeepONet-type\2d-singular\extra_data.npz", idx_b=idx_b, coeff_b=coeff_b,
    #          rhs_b=rhs_b, idx_i=idx_i, coeff_i=coeff_i, rhs_i=rhs_i)

    
    
