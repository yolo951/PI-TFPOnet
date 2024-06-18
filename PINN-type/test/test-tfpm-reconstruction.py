
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy
from scipy.integrate import quad
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

def compute_params(a, b, x, x1, x2, y1, y2, f_A_default, f_B_default, f_F_default, F):
    y = a * x +  b
    if abs(a) < 1e-8:
        if abs(b) < 1e-8:
            lamda, mu = 1.0, x1
            F_val = quad(integrand_linear, x1, x2, args=(x, F))[0]
            G_val = -quad(F, x1, x2)[0]
            gamma, delta = 0.0, 1.0
            lamda_p, mu_p, lamda_pp, mu_pp = 0.0, 1.0, 0.0, 0.0
            f_A = lambda x: np.where((x1<x) & (x<=x2), 1.0, f_A_default(x))
            f_B = lambda x: np.where((x1<x) & (x<=x2), x, f_B_default(x))
            f_F = lambda x: np.where((x1<x) & (x<=x2), quad(integrand_linear, x1, x2, args=(x, F))[0], f_F_default(x))
        else:
            sqrt_b = np.sqrt(b)
            lamda, mu = np.exp(sqrt_b * x), np.exp(-sqrt_b * x)
            F_val = quad(integrand_sinh, x1, x2, args=(x, F, b))[0]
            G_val = quad(integrandp_sinh, x1, x2, args=(x, F, b))[0]
            gamma, delta = sqrt_b * lamda, -sqrt_b * mu
            lamda_p, mu_p = sqrt_b * np.exp(x * sqrt_b), -sqrt_b * np.exp(-x * sqrt_b)
            lamda_pp, mu_pp = b * np.exp(x * sqrt_b), b * np.exp(-x * sqrt_b)
            f_A = lambda x: np.where((x1<x) & (x<=x2), np.exp(sqrt_b * x), f_A_default(x))
            f_B = lambda x: np.where((x1<x) & (x<=x2), np.exp(-sqrt_b * x), f_B_default(x))
            f_F = lambda x: np.where((x1<x) & (x<=x2), quad(integrand_sinh, x1, x2, args=(x, F, b))[0], f_F_default(x))
    else:
        z, z1, z2 = y * np.power(np.abs(a), -2 / 3), y1 * np.power(np.abs(a), -2 / 3), y2 * np.power(np.abs(a), -2 / 3)
        lamda, gamma, mu, delta = airy(z)
        gamma, delta = gamma * np.cbrt(a), delta * np.cbrt(a)
        F_val = quad(integrand_airy, z1, z2, args=(z, F, a, b))[0] if z2 >= z1 else quad(integrand_airy, z2, z1, args=(z, F, a, b))[0]
        G_val = quad(integrandp_airy, z1, z2, args=(z, F, a, b))[0] if z2 >= z1 else quad(integrandp_airy, z2, z1, args=(z, F, a, b))[0]
        lamda_p, mu_p = airy(z)[1] * np.cbrt(a), airy(z)[3] * np.cbrt(a)
        lamda_pp, mu_pp = z * lamda * np.cbrt(a)**2, z * mu * np.cbrt(a)**2
        f_z = lambda x: (a*x+b) * np.power(np.abs(a), -2 / 3)
        f_A = lambda x: np.where((x1<x) & (x<=x2), airy(f_z(x))[0], f_A_default(x))
        f_B = lambda x: np.where((x1<x) & (x<=x2), airy(f_z(x))[2], f_B_default(x))
        f_F_x = lambda x: quad(integrand_airy, z1, z2, args=(f_z(x), F, a, b))[0] if z2 >= z1 else quad(integrand_airy, z2, z1, args=(f_z(x), F, a, b))[0]
        f_F = lambda x: np.where((x1<x) & (x<=x2), f_F_x(x), f_F_default(x))
    if x == grid[0]:
        f_A = lambda x: np.where(x==x1, lamda, f_A_default(x))
        f_B = lambda x: np.where(x==x1, mu, f_B_default(x))
        f_F = lambda x: np.where(x==x1, F_val, f_F_default(x))
    
    return lamda, gamma, mu, delta, F_val, G_val, lamda_p, mu_p, lamda_pp, mu_pp, f_A, f_B, f_F

# solve equation -u''(x)+q(x)u(x)=F(x)
def tfpm(grid, qLeft, qRight, F):
    N = len(grid)
    U = np.zeros((2*(N-1), 2*(N-1)), dtype=np.float64)
    B = np.zeros((2*(N-1),), dtype=np.float64)
    x1 = np.zeros((int((N+1)/2)), dtype=np.float64)
    x2 = np.zeros((int((N + 1) / 2)), dtype=np.float64)
    f_A = lambda x: x
    f_B = lambda x: x
    f_F = lambda x: x
    a, b = compute_coefficients(grid[0], grid[1], qRight[0], qLeft[1])
    boundary_params = compute_params(a, b, grid[0], grid[0], grid[1], qRight[0], qLeft[1], f_A, f_B, f_F, F)
    lambda1, gamma1, mu1, delta1, F1, G1, lambda1p, mu1p, lambda1pp, mu1pp, f_A, f_B, f_F = boundary_params
    U[0,:2] = np.array([-lambda1, -mu1])
    B[0] = F1
    coeff = [[lambda1, mu1, F1]]
    coeffp = [[lambda1p, mu1p]]
    coeffpp = [[lambda1pp, mu1pp]]

    for i in range(1, 2*N-3, 2):
        a1, b1 = compute_coefficients(grid[i // 2], grid[i // 2 + 1], qRight[i // 2], qLeft[i // 2 + 1])
        a2, b2 = compute_coefficients(grid[i // 2 + 1], grid[i // 2 + 2], qRight[i // 2 + 1], qLeft[i // 2 + 2])
        
        params1 = compute_params(a1, b1, grid[i // 2 + 1], grid[i // 2], grid[i // 2 + 1], qRight[i // 2], qLeft[i // 2 + 1], f_A, f_B, f_F, F)
        lambda1, gamma1, mu1, delta1, F1, G1, lambda1p, mu1p, lambda1pp, mu1pp, f_A, f_B, f_F = params1
        params2 = compute_params(a2, b2, grid[i // 2 + 1], grid[i // 2 + 1], grid[i // 2 + 2], qRight[i // 2 + 1], qLeft[i // 2 + 2], f_A, f_B, f_F, F)
        lambda2, gamma2, mu2, delta2, F2, G2, lambda2p, mu2p, lambda2pp, mu2pp, f_A, f_B, f_F = params2

        B[i] = F2 - F1
        B[i+1] = G2 - G1
        U[i, i-1:i+3] = np.array([lambda1, mu1, -lambda2, -mu2])
        U[i+1, i - 1:i + 3] = np.array([gamma1, delta1, -gamma2, -delta2])
        coeff.append([lambda1, mu1, F1])
        coeffp.append([gamma1, delta1])
        coeffpp.append([lambda1pp, mu1pp])

    a, b = compute_coefficients(grid[-2], grid[-1], qRight[-2], qLeft[-1])
    boundary_params = compute_params(a, b, grid[-1], grid[-2], grid[-1], qRight[-2], qLeft[-1], f_A, f_B, f_F, F)
    lambda1, gamma1, mu1, delta1, F1, G1, lambda1p, mu1p, lambda1pp, mu1pp, f_A, f_B, f_F = boundary_params

    U[2*N-3, -2:] = np.array([-lambda1, -mu1])
    B[2*N-3] = F1
    B[N-2] -= 1.0
    B[N-1] -= 1.0
    coeff.append([lambda1, mu1, F1])
    coeffp.append([lambda1p, mu1p])
    coeffpp.append([lambda1pp, mu1pp])
    
    AB = np.linalg.solve(U, B).flatten()
    def f_AB(x):
        result = np.array([AB[0], AB[1]])
        for i in range(1, N):
            result = np.where((grid[i-1] < x) & (x <= grid[i]), AB[2*(i-1):2*(i-1)+2], result)
        return result
    x = np.zeros(N)
    for i in range(N):
        x[i] = f_AB(grid[i])[0]*f_A(grid[i])+f_AB(grid[i])[1]*f_B(grid[i])+f_F(grid[i])
        # x[i] = AB[2*(i-1)]*coeff[i][0]+AB[2*(i-1)+1]*coeff[i][1]+coeff[i][2]
    x1[:] = x[:int((N+1)/2)]
    x2[:] = x[int((N+1)/2)-1:]
    x2[0] += 1
    return x1, x2, U, B, coeff, f_AB, coeffp, coeffpp, f_A, f_B, f_F

N = 11
q1 = lambda x: 5.0
q2 = lambda x: 0.1*(4+32*x)
F = lambda x: x

q = lambda x: np.where(x<=0.5, q1(x), q2(x))
grid = np.linspace(0, 1, N)
qLeft, qRight = q(grid), q(grid)
qRight[int((N-1)/2)] = q2(grid[int((N-1)/2)])
u1, u2, U, B, coeff, f_AB, coeffp, coeffpp, f_A, f_B, f_F = tfpm(grid, qLeft, qRight, F)

N_fine = 101
grid_fine = np.linspace(0, 1, N_fine)
x = np.zeros(N_fine)
x1 = np.zeros((int((N_fine+1)/2)), dtype=np.float64)
x2 = np.zeros((int((N_fine + 1) / 2)), dtype=np.float64)
for i in range(N_fine):
    x[i] = f_AB(grid_fine[i])[0]*f_A(grid_fine[i])+f_AB(grid_fine[i])[1]*f_B(grid_fine[i])+f_F(grid_fine[i])
    # x[i] = AB[2*(i-1)]*coeff[i][0]+AB[2*(i-1)+1]*coeff[i][1]+coeff[i][2]
x1[:] = x[:int((N_fine+1)/2)]
x2[:] = x[int((N_fine+1)/2)-1:]
x2[0] += 1

q = lambda x: np.where(x<=0.5, q1(x), q2(x))
grid = np.linspace(0, 1, N_fine)
qLeft, qRight = q(grid), q(grid)
qRight[int((N_fine-1)/2)] = q2(grid[int((N_fine-1)/2)])
u1, u2, U, B, coeff, f_AB, coeffp, coeffpp, f_A, f_B, f_F = tfpm(grid, qLeft, qRight, F)

plt.plot(grid_fine[:int(N_fine/2+1)], x1, 'b-', alpha=1., zorder=0, label='reconstruction')
plt.plot(grid_fine[int(N_fine/2+1):], x2[1:], 'b-', alpha=1., zorder=0)
plt.plot(grid_fine[:int(N_fine/2+1)], u1, 'r-', alpha=1., zorder=0, label='tfpm')
plt.plot(grid_fine[int(N_fine/2+1):], u2[1:], 'r-', alpha=1., zorder=0)
plt.legend()
plt.show()
