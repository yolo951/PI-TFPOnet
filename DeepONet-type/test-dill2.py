

import sys
sys.path.append('D:\pythonProject\TFP-Net\DeepONet-type')
import torch
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy
from scipy.integrate import quad
from math import sqrt
from scipy import interpolate
from generate_f import generate
import time
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_coefficients(x1, x2, y1, y2):
    c = np.polyfit([x1, x2], [y1, y2], 1)
    return c[0], c[1]

def integrand_linear(s, y, F):
    G = np.where(y >= s, s - y, 0)
    return F(s) * G

def integrand_sinh(s, y, F, b):
    G = 1 / (2 * np.sqrt(b)) * np.where(y >= s, np.sinh(np.sqrt(b) * (s - y)), np.sinh(np.sqrt(b) * (y - s)))
    return F(s) * G

def integrandp_sinh(s, y, F, b):
    G = 1 / 2 * np.where(y > s, -np.cosh(np.sqrt(b) * (s - y)), np.where(y < s, np.cosh(np.sqrt(b) * (y - s)), 0.0))
    return F(s) * G

def integrand_airy(s, z, F, a, b):
    G = np.pi / 2 * np.where(z >= s, airy(s)[2] * airy(z)[0] - airy(s)[0] * airy(z)[2], airy(s)[0] * airy(z)[2] - airy(s)[2] * airy(z)[0])
    return np.power(np.abs(a), -2 / 3) * G * F(s / np.cbrt(a) - b / a)

def integrandp_airy(s, z, F, a, b):
    G = np.pi / 2 * np.where(z > s, airy(s)[2] * airy(z)[1] - airy(s)[0] * airy(z)[3], np.where(z < s, airy(s)[0] * airy(z)[3] - airy(s)[2] * airy(z)[1], 0.0))
    return 1 / np.cbrt(a) * G * F(s / np.cbrt(a) - b / a)

def compute_params(a, b, x, x1, x2, y1, y2, f):
    interpolate_f = interpolate.interp1d(np.linspace(0, 1, f.shape[-1]), f)
    F = lambda x: interpolate_f(x)
    y = a * x +  b
    if abs(a) < 1e-8:
        if abs(b) < 1e-8:
            lamda, mu = 1.0, x1
            F_val = quad(integrand_linear, x1, x2, args=(x, F))[0]
            G_val = -quad(F, x1, x2)[0]
            gamma, delta = 0.0, 1.0
            lamda_p, mu_p, lamda_pp, mu_pp = 0.0, 1.0, 0.0, 0.0
            f_A = lambda x: 1.0
            f_B = lambda x: x
            f_rhs = lambda x: quad(integrand_linear, x1, x2, args=(x, F))[0]
        else:
            sqrt_b = np.sqrt(b)
            lamda, mu = np.exp(sqrt_b * x), np.exp(-sqrt_b * x)
            F_val = quad(integrand_sinh, x1, x2, args=(x, F, b))[0]
            G_val = quad(integrandp_sinh, x1, x2, args=(x, F, b))[0]
            gamma, delta = sqrt_b * lamda, -sqrt_b * mu
            lamda_p, mu_p = sqrt_b * np.exp(x * sqrt_b), -sqrt_b * np.exp(-x * sqrt_b)
            lamda_pp, mu_pp = b * np.exp(x * sqrt_b), b * np.exp(-x * sqrt_b)
            f_A = lambda x: np.exp(sqrt_b * x)
            f_B = lambda x: np.exp(-sqrt_b * x)
            f_rhs = lambda x: quad(integrand_sinh, x1, x2, args=(x, F, b))[0]
    else:
        z, z1, z2 = y * np.power(np.abs(a), -2 / 3), y1 * np.power(np.abs(a), -2 / 3), y2 * np.power(np.abs(a), -2 / 3)
        lamda, gamma, mu, delta = airy(z)
        gamma, delta = gamma * np.cbrt(a), delta * np.cbrt(a)
        F_val = quad(integrand_airy, z1, z2, args=(z, F, a, b))[0] if z2 >= z1 else quad(integrand_airy, z2, z1, args=(z, F, a, b))[0]
        G_val = quad(integrandp_airy, z1, z2, args=(z, F, a, b))[0] if z2 >= z1 else quad(integrandp_airy, z2, z1, args=(z, F, a, b))[0]
        lamda_p, mu_p = airy(z)[1] * np.cbrt(a), airy(z)[3] * np.cbrt(a)
        lamda_pp, mu_pp = z * lamda * np.cbrt(a)**2, z * mu * np.cbrt(a)**2
        f_z = lambda x: (a*x+b) * np.power(np.abs(a), -2 / 3)
        f_A = lambda x: airy(f_z(x))[0]
        f_B = lambda x: airy(f_z(x))[2]
        f_rhs = lambda x: quad(integrand_airy, z1, z2, args=(f_z(x), F, a, b))[0] if z2 >= z1 else quad(integrand_airy, z2, z1, args=(f_z(x), F, a, b))[0]

    
    return lamda, gamma, mu, delta, F_val, G_val, lamda_p, mu_p, lamda_pp, mu_pp, f_A, f_B, f_rhs

# solve equation -u''(x)+q(x)u(x)=F(x)
def tfpm(grid, qLeft, qRight, F):
    N = len(grid)
    K = len(F)
    x1 = np.zeros((K, int((N+1)/2)), dtype=np.float64)
    x2 = np.zeros((K, int((N + 1) / 2)), dtype=np.float64)
    B = np.zeros((K, 2*(N-1)), dtype=np.float64)
    all_f_rhs = []
    all_f_AB = []
    
    for k in range(K):
        U = np.zeros((2*(N-1), 2*(N-1)), dtype=np.float64)
        each_B = np.zeros((2*(N-1), ), dtype=np.float64)
        each_f_A = []
        each_f_B = []
        each_f_rhs = []
        a, b = compute_coefficients(grid[0], grid[1], qRight[0], qLeft[1])
        boundary_params = compute_params(a, b, grid[0], grid[0], grid[1], qRight[0], qLeft[1], f[k])
        lambda1, gamma1, mu1, delta1, F1, G1, lambda1p, mu1p, lambda1pp, mu1pp, f_A, f_B, f_rhs = boundary_params
        each_f_A.append(f_A)
        each_f_B.append(f_B)
        each_f_rhs.append(f_rhs)
        U[0,:2] = np.array([-lambda1, -mu1])
        each_B[0] = F1

        for i in range(1, 2*N-3, 2):
            a1, b1 = compute_coefficients(grid[i // 2], grid[i // 2 + 1], qRight[i // 2], qLeft[i // 2 + 1])
            a2, b2 = compute_coefficients(grid[i // 2 + 1], grid[i // 2 + 2], qRight[i // 2 + 1], qLeft[i // 2 + 2])
            
            params1 = compute_params(a1, b1, grid[i // 2 + 1], grid[i // 2], grid[i // 2 + 1], qRight[i // 2], qLeft[i // 2 + 1], f[k])
            lambda1, gamma1, mu1, delta1, F1, G1, lambda1p, mu1p, lambda1pp, mu1pp, f_A, f_B, f_rhs = params1
            each_f_A.append(f_A)
            each_f_B.append(f_B)
            each_f_rhs.append(f_rhs)
            params2 = compute_params(a2, b2, grid[i // 2 + 1], grid[i // 2 + 1], grid[i // 2 + 2], qRight[i // 2 + 1], qLeft[i // 2 + 2], f[k])
            lambda2, gamma2, mu2, delta2, F2, G2, lambda2p, mu2p, lambda2pp, mu2pp, f_A, f_B, f_rhs = params2
            each_f_A.append(f_A)
            each_f_B.append(f_B)
            each_f_rhs.append(f_rhs)

            each_B[i] = F2 - F1
            each_B[i+1] = G2 - G1
            U[i, i-1:i+3] = np.array([lambda1, mu1, -lambda2, -mu2])
            U[i+1, i - 1:i + 3] = np.array([gamma1, delta1, -gamma2, -delta2])

        a, b = compute_coefficients(grid[-2], grid[-1], qRight[-2], qLeft[-1])
        boundary_params = compute_params(a, b, grid[-1], grid[-2], grid[-1], qRight[-2], qLeft[-1], f[k])
        lambda1, gamma1, mu1, delta1, F1, G1, lambda1p, mu1p, lambda1pp, mu1pp, f_A, f_B, f_rhs = boundary_params
        each_f_A.append(f_A)
        each_f_B.append(f_B)
        each_f_rhs.append(f_rhs)

        U[2*N-3, -2:] = np.array([-lambda1, -mu1])
        each_B[2*N-3] = F1
        each_B[N-2] -= 1.0
        each_B[N-1] -= 1.0

        
        M_inverse = np.diag(1 / np.max(np.abs(U), axis=0))
        M_inverse = np.eye(U.shape[0])

        scaled_U = np.matmul(U, M_inverse)

        AB = np.linalg.solve(scaled_U, each_B).flatten() # scaled AB
        scaled_AB = np.linalg.solve(scaled_U, each_B).flatten()
        scale = np.diag(M_inverse)
        scale1 = scale[::2]
        scale2 = scale[1::2]
        scaled_f_A, scaled_f_B, f_AB = [], [], []
        for i in range(1, N):
            f_AB.append(scaled_AB[2*(i-1):2*(i-1)+2])
            scaled_f_A.extend([lambda x: each_f_A[i](x)*scale1[i-1]])
            scaled_f_B.extend([lambda x: each_f_B[i](x)*scale2[i-1]])
        x = np.zeros(N)
        for i in range(N):
            idx = 0 if i==0 else i-1
            x[i] = f_AB[idx][0]*scaled_f_A[idx](grid[i])+f_AB[idx][1]*scaled_f_B[idx](grid[i])+each_f_rhs[idx](grid[i])
            # x[i] = AB[2*(i-1)]*coeff[i][0]+AB[2*(i-1)+1]*coeff[i][1]+coeff[i][2]
        x1[k, :] = x[:int((N+1)/2)]
        x2[k, :] = x[int((N+1)/2)-1:]
        x2[k, 0] += 1
        B[k, :] = each_B[:]
        all_f_rhs.append(each_f_rhs)
        all_f_AB.append(f_AB)
    
    return x1, x2, scaled_U, B, all_f_AB, scaled_f_A, scaled_f_B, all_f_rhs


N = 17
q1 = lambda x: 1+np.exp(x)
q2 = lambda x: 1-np.log(x+1)
f = generate(samples=1)  # 前100个作为训练，最后一个做测试
grid = np.linspace(0, 1, f.shape[-1])

q = lambda x: np.where(x<=0.5, q1(x), q2(x))
grid = np.linspace(0, 1, N)
qLeft, qRight = q(grid), q(grid)
qRight[int((N-1)/2)] = q2(grid[int((N-1)/2)])

N_f = N
grid_f = np.linspace(0, 1, N_f)
u1, u2, U, B, f_AB, f_A, f_B, f_rhs = tfpm(grid, qLeft, qRight, f)

fig = plt.figure(figsize=(4, 3), dpi=150)

plt.plot(grid[:int(N/2+1)], u1.flatten(), 'r-', label='Ground Truth', alpha=1., zorder=0)
plt.plot(grid[int(N/2+1):], u2[:, 1:].flatten(), 'r-', alpha=1., zorder=0)
plt.legend(loc='best')
plt.xlabel("x")
plt.ylabel("$u$")
plt.grid()

plt.tight_layout()
plt.show(block=True)
