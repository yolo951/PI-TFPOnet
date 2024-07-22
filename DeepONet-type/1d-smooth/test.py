

import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import interpolate
import dill
from collections import OrderedDict
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def make_function(f):
    return lambda x: f(x)

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
            f_rhs = lambda x, x1=x1, x2=x2, F=F: quad(integrand_linear, x1, x2, args=(x, F))[0]
        else:
            sqrt_b = np.sqrt(b)
            lamda, mu = np.exp(sqrt_b * x), np.exp(-sqrt_b * x)
            F_val = quad(integrand_sinh, x1, x2, args=(x, F, b))[0]
            G_val = quad(integrandp_sinh, x1, x2, args=(x, F, b))[0]
            gamma, delta = sqrt_b * lamda, -sqrt_b * mu
            lamda_p, mu_p = sqrt_b * np.exp(x * sqrt_b), -sqrt_b * np.exp(-x * sqrt_b)
            lamda_pp, mu_pp = b * np.exp(x * sqrt_b), b * np.exp(-x * sqrt_b)
            f_A = lambda x, sqrt_b=sqrt_b: np.exp(sqrt_b * x)
            f_B = lambda x, sqrt_b=sqrt_b: np.exp(-sqrt_b * x)
            f_rhs = lambda x, x1=x1, x2=x2, F=F, b=b: quad(integrand_sinh, x1, x2, args=(x, F, b))[0]
    else:
        z, z1, z2 = y * np.power(np.abs(a), -2 / 3), y1 * np.power(np.abs(a), -2 / 3), y2 * np.power(np.abs(a), -2 / 3)
        lamda, gamma, mu, delta = airy(z)
        gamma, delta = gamma * np.cbrt(a), delta * np.cbrt(a)
        F_val = quad(integrand_airy, z1, z2, args=(z, F, a, b))[0] if z2 >= z1 else quad(integrand_airy, z2, z1, args=(z, F, a, b))[0]
        G_val = quad(integrandp_airy, z1, z2, args=(z, F, a, b))[0] if z2 >= z1 else quad(integrandp_airy, z2, z1, args=(z, F, a, b))[0]
        lamda_p, mu_p = airy(z)[1] * np.cbrt(a), airy(z)[3] * np.cbrt(a)
        lamda_pp, mu_pp = z * lamda * np.cbrt(a)**2, z * mu * np.cbrt(a)**2
        f_z = lambda x, a=a, b=b: (a*x+b) * np.power(np.abs(a), -2 / 3)
        f_A = lambda x: airy(f_z(x))[0]
        f_B = lambda x: airy(f_z(x))[2]
        f_rhs = lambda x, z1=z1, z2=z2, F=F, a=a, b=b: quad(integrand_airy, z1, z2, args=(f_z(x), F, a, b))[0] if z2 >= z1 else quad(integrand_airy, z2, z1, args=(f_z(x), F, a, b))[0]

    
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
        each_f_A.extend([lambda x, lambda1=lambda1: lambda1])
        each_f_B.extend([lambda x, mu=mu1: mu])
        each_f_rhs.extend([lambda x, F=F1: F])
        U[0,:2] = np.array([-lambda1, -mu1])
        each_B[0] = F1

        for i in range(1, 2*N-3, 2):
            a1, b1 = compute_coefficients(grid[i // 2], grid[i // 2 + 1], qRight[i // 2], qLeft[i // 2 + 1])
            a2, b2 = compute_coefficients(grid[i // 2 + 1], grid[i // 2 + 2], qRight[i // 2 + 1], qLeft[i // 2 + 2])
            
            params1 = compute_params(a1, b1, grid[i // 2 + 1], grid[i // 2], grid[i // 2 + 1], qRight[i // 2], qLeft[i // 2 + 1], f[k])
            lambda1, gamma1, mu1, delta1, F1, G1, lambda1p, mu1p, lambda1pp, mu1pp, f_A, f_B, f_rhs = params1
            each_f_A.append(make_function(f_A))
            each_f_B.append(make_function(f_B))
            each_f_rhs.append(make_function(f_rhs))
            params2 = compute_params(a2, b2, grid[i // 2 + 1], grid[i // 2 + 1], grid[i // 2 + 2], qRight[i // 2 + 1], qLeft[i // 2 + 2], f[k])
            lambda2, gamma2, mu2, delta2, F2, G2, lambda2p, mu2p, lambda2pp, mu2pp, f_A, f_B, f_rhs = params2

            each_B[i] = F2 - F1
            each_B[i+1] = G2 - G1
            U[i, i-1:i+3] = np.array([lambda1, mu1, -lambda2, -mu2])
            U[i+1, i - 1:i + 3] = np.array([gamma1, delta1, -gamma2, -delta2])

        a, b = compute_coefficients(grid[-2], grid[-1], qRight[-2], qLeft[-1])
        boundary_params = compute_params(a, b, grid[-1], grid[-2], grid[-1], qRight[-2], qLeft[-1], f[k])
        lambda1, gamma1, mu1, delta1, F1, G1, lambda1p, mu1p, lambda1pp, mu1pp, f_A, f_B, f_rhs = boundary_params
        each_f_A.append(make_function(f_A))
        each_f_B.append(make_function(f_B))
        each_f_rhs.append(make_function(f_rhs))

        U[2*N-3, -2:] = np.array([-lambda1, -mu1])
        each_B[2*N-3] = F1
        each_B[N-2] -= 1.0
        each_B[N-1] -= 1.0

        
        M_inverse = np.diag(1 / np.max(np.abs(U), axis=0))
        # M_inverse = np.eye(U.shape[0])

        scaled_U = np.matmul(U, M_inverse)
        each_B = each_B

        AB = np.linalg.solve(scaled_U, each_B).flatten() # scaled AB
        scaled_AB = np.linalg.solve(scaled_U, each_B).flatten()
        scale = np.diag(M_inverse)
        scale1 = scale[::2]
        scale2 = scale[1::2]
        scaled_f_A, scaled_f_B, f_AB = [lambda x, scale1_0=scale1[0]: each_f_A[0](x)*scale1_0], [lambda x, scaled2_0=scale2[0]: each_f_B[0](x)*scaled2_0], [[scaled_AB[0], scaled_AB[1]]]
        for i in range(1, N):
            f_AB.append(scaled_AB[2*(i-1):2*(i-1)+2])
            scaled_f_A.extend([lambda x, i=i, scale1_=scale1[i-1]: each_f_A[i](x)*scale1_])
            scaled_f_B.extend([lambda x, i=i, scale2_=scale2[i-1]: each_f_B[i](x)*scale2_])
        x = np.zeros(N)
        for i in range(N):
            x[i] = f_AB[i][0] * scaled_f_A[i](grid[i]) + f_AB[i][1] * scaled_f_B[i](grid[i]) + each_f_rhs[i](grid[i])
        x1[k, :] = x[:int((N+1)/2)]
        x2[k, :] = x[int((N+1)/2)-1:]
        x2[k, 0] += 1
        B[k, :] = each_B[:]
        all_f_rhs.append(copy.deepcopy(each_f_rhs))
        all_f_AB.append(copy.deepcopy(f_AB))
    
    return x1, x2, scaled_U, B, all_f_AB, scaled_f_A, scaled_f_B, all_f_rhs
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        
        self.depth = len(layers) - 1
        self.activation = torch.nn.ReLU
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x):
        out = self.layers(x)
        return out

def recovery(x, f_A, f_B, f_rhs, f_AB):
    K = len(f_AB)
    recovery_u = np.zeros((K, len(x)))
    h = 1/(len(f_A)-1)
    for k in  range(K):
        for i in range(len(x)):
            idx = int((x[i]-h**2)//h)+1 if x[i]>0 else 0
            recovery_u[k, i] = f_AB[k][idx][0]*f_A[idx](x[i])+f_AB[k][idx][1]*f_B[idx](x[i])+f_rhs[k][idx](x[i])
    return recovery_u

# loss = np.load('loss_history.npy')
# epochs = len(loss)
# test_error = np.load('test_error_history.npy')
# fig = plt.figure(figsize=(4, 3), dpi=150)
# plt.plot(np.arange(0, epochs), loss)
# plt.yscale("log")
# plt.xlabel("epochs")
# plt.tight_layout()
# plt.savefig('1d_smooth_loss')


# fig = plt.figure(figsize=(4, 3), dpi=150)
# plt.plot(np.arange(0, epochs+1, 100), test_error, '-*')
# plt.yscale("log")
# plt.xlabel("epochs")
# plt.tight_layout()
# plt.savefig('1d_smooth_error')
# plt.show()

N_f = N = 33
N_fine = 257
ntrain, ntest = 1000, 200
layers = [N_f, 16, 32, 32, 2*N]
model = DNN(layers).to(device)
model.load_state_dict(torch.load('model.pt'))
grid_fine = np.linspace(0, 1, N_fine)
f = np.load('f.npy')
f_AB = np.load('f_AB.npy')
with open('f_A.pkl', 'rb') as ff:
    f_A = dill.load(ff)
with open('f_B.pkl', 'rb') as ff:
    f_B = dill.load(ff)
with open('f_rhs.pkl', 'rb') as ff:
    f_rhs = dill.load(ff)
prediction = model.predict(grid_fine, f_A, f_B, f_rhs[-ntest:], f[-ntest:], f_AB=f_AB[-ntest:])
u_test = recovery(grid_fine, f_A, f_B, f_rhs[-ntest:], f_AB[-ntest:])
errors = np.abs(prediction-u_test)

fig = plt.figure(figsize=(4, 3), dpi=150)
for i in range(ntest):
    plt.plot(grid_fine, errors[i], color='gray', linestyle='--')
plt.yscale("log")
plt.xlabel("x")
plt.tight_layout()
plt.savefig('1d_smooth_errors')

u_test = grid_fine[:]
prediction = grid_fine*1.2
fig = plt.figure(figsize=(4, 3), dpi=150)
plt.plot(grid_fine[:int(N_fine/2+1)], u_test[-ntest, :int(N_fine/2+1)].flatten(), 'b-', label='Ground Truth', linewidth=2, alpha=1., zorder=0)
plt.plot(grid_fine[int(N_fine/2+1):], u_test[-ntest, int(N_fine/2+1):].flatten(), 'b-', linewidth=2, alpha=1., zorder=0)
plt.plot(grid_fine[:int(N_fine/2+1)], prediction[-ntest, :int(N_fine/2+1)], 'r--', label='Prediction', linewidth=2, alpha=1., zorder=0)
plt.plot(grid_fine[int(N_fine/2+1):], prediction[-ntest, int(N_fine/2+1):], 'r--', linewidth=2, alpha=1., zorder=0)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('1d_smooth_example')
plt.show()

