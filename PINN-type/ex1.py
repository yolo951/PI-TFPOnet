
import torch
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy
from scipy.integrate import quad
from math import sqrt
from scipy import interpolate
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

    a, b = compute_coefficients(grid[-2], grid[-1], qRight[-2], qLeft[-1])
    boundary_params = compute_params(a, b, grid[-1], grid[-2], grid[-1], qRight[-2], qLeft[-1], f_A, f_B, f_F, F)
    lambda1, gamma1, mu1, delta1, F1, G1, lambda1p, mu1p, lambda1pp, mu1pp, f_A, f_B, f_F = boundary_params

    U[2*N-3, -2:] = np.array([-lambda1, -mu1])
    B[2*N-3] = F1
    B[N-2] -= 1.0
    B[N-1] -= 1.0
    
    M_inverse = np.diag(1 / np.max(np.abs(U), axis=0))
    # M_inverse = np.eye(U.shape[0])
    scaled_U = np.matmul(U, M_inverse)
    AB = np.linalg.solve(scaled_U, B).flatten() # scaled AB
    start_time = time.time()
    scaled_AB = np.linalg.solve(scaled_U, B).flatten()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Solving Ux=B time: {elapsed_time:.6f} seconds")
    def f_AB(x):
        result = np.array([scaled_AB[0], scaled_AB[1]])
        for i in range(1, N):
            result = np.where((grid[i-1] < x) & (x <= grid[i]), scaled_AB[2*(i-1):2*(i-1)+2], result)
        return result
    scale = np.diag(M_inverse)
    scale1 = scale[::2]
    scale2 = scale[1::2]
    def scaled_f_A(x):
        result_f_A = f_A(grid[0])*scale1[0]
        for i in range(1, N):
            result_f_A = np.where((grid[i-1] < x) & (x <= grid[i]), f_A(x)*scale1[i-1], result_f_A)
        return result_f_A
    def scaled_f_B(x):
        result_f_B = f_B(grid[0])*scale2[0]
        for i in range(1, N):
            result_f_B = np.where((grid[i-1] < x) & (x <= grid[i]), f_B(x)*scale2[i-1], result_f_B)
        return result_f_B
    x = np.zeros(N)
    for i in range(N):
        x[i] = f_AB(grid[i])[0]*scaled_f_A(grid[i])+f_AB(grid[i])[1]*scaled_f_B(grid[i])+f_F(grid[i])
        # x[i] = AB[2*(i-1)]*coeff[i][0]+AB[2*(i-1)+1]*coeff[i][1]+coeff[i][2]
    x1[:] = x[:int((N+1)/2)]
    x2[:] = x[int((N+1)/2)-1:]
    x2[0] += 1
    return x1, x2, scaled_U, B, f_AB, scaled_f_A, scaled_f_B, f_F


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


class PhysicsInformedNN():
    def __init__(self, X, U, B, layers):
        
        self.x = torch.tensor(X).float().to(device)
        U = torch.tensor(U).float().to(device)
        B = torch.tensor(B).float().to(device)

        N = int(U.shape[0]/2)
        self.U = torch.zeros((2*N-2, 2*N+2)).float().to(device)
        self.U_boundary = torch.zeros((2, 2*N+2)).float().to(device)
        self.U_interface = torch.zeros((2, 2*N+2)).float().to(device)
        self.U[2:, 2:] = torch.cat((U[1:N-1,:], U[N+1:-1,:]), dim=0)
        self.U[0, :3] = torch.tensor([1.0, 0.0, -1.0])
        self.U[1, 1:4] = torch.tensor([1.0, 0.0, -1.0])
        self.U_boundary[0, :-2] = U[0,:]
        self.U_boundary[-1, 2:] = U[-1,:]
        self.U_interface[:, 2:] = U[N-1:N+1, :]
        self.B = torch.zeros((2*N-2)).float().to(device)
        self.B[2:] = torch.cat((B[1:N-1], B[N+1:-1]), dim=0)
        self.B_interface = B[N-1:N+1]
        self.B_boundary = torch.stack((B[0], B[-1]))
        # self.U = torch.cat((U[1:N-1,:], U[N+1:-1,:]), dim=0)
        # self.U_boundary = torch.stack((U[0,:], U[-1,:]))
        # self.U_interface = U[N-1:N+1, :]
        # self.B = torch.cat((B[1:N-1], B[N+1:-1]), dim=0)
        # self.B_interface = B[N-1:N+1]
        # self.B_boundary = torch.stack((B[0], B[-1]))
        self.dnn = DNN(layers).to(device)
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0, 
            max_iter=50000, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-15, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=2e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.iter = 0
        self.loss = torch.nn.MSELoss()
        self.loss_history = []
        self.lossContinuos_history = []
        self.lossJump_history = []
        self.lossBoundary_history = []
        self.gamma = 100.0
        self.gamma_interface = 1.0
        self.gamma_boundary = 100.0
        self.gamma_equ = 10.0
        # self.coeff = torch.tensor(coeff).float().to(device)
        # self.coeffpp = torch.tensor(coeffpp).float().to(device)

    # def loss_equ(self):
    #     q = lambda x: torch.where(x<=0.5, 5.0, 0.1*(4+32*x))
    #     qh = lambda x: torch.where(x<=0.5, 5.0, 0.1*(4+32*x))
    #     f = lambda x: x
    #     AB = self.dnn(self.x)
    #     u0 = torch.sum(self.coeff[1:-1, :2] * AB[:-1], dim=1)
    #     upp0 = torch.sum(self.coeffpp[1:-1, :2] * AB[:-1], dim=1)
    #     # FG = \int F*G, -FG''+qh*FG=F
    #     # then -FG''+q*FG = -FG''+qh*FG+(q-qh)*FG=F+(q-qh)*FG
    #     loss = self.loss(upp0, q(self.x[1:-1]).flatten()*u0) + \
    #         self.loss(qh(self.x[1:-1]).flatten()*self.coeff[1:-1, 2], q(self.x[1:-1]).flatten()*self.coeff[1:-1, 2])
    #     # loss = self.loss(-upp+q(self.x[:-1]).flatten()*u, f(self.x[:-1]).flatten())
    #     return loss

    # def loss_func(self):
    #     self.optimizer.zero_grad()
    #     AB_pred = self.dnn(self.x)
    #     AB_pred = AB_pred.flatten()
    #     loss = self.gamma*self.loss(torch.matmul(self.U, AB_pred), self.B)\
                    # +self.gamma_boundary*self.loss(torch.matmul(self.U_boundary, AB_pred), self.B_boundary)\
                    #   + self.gamma_interface*self.loss(torch.matmul(self.U_interface, AB_pred), self.B_interface)
    #     loss += self.gamma_equ*self.loss_equ()
    #     loss.backward()

    #     self.loss_history.append(loss.item())
    #     self.iter += 1
    #     if (self.iter+1) % 1 == 0:
    #         print(f'LBFGS optimizer {self.iter}th Loss {loss.item()}')
    #     return loss
    
    def train(self, nIter1, nIter):
        self.dnn.train()
        for epoch in range(nIter):
            self.optimizer.zero_grad()
            AB_pred = self.dnn(self.x)
            AB_pred = AB_pred.flatten()
            if epoch <= nIter1:
                loss = self.loss_equ()
            else:
                lossContinuous = self.gamma*self.loss(torch.matmul(self.U, AB_pred), self.B)
                lossBoundary = self.gamma_boundary*self.loss(torch.matmul(self.U_boundary, AB_pred), self.B_boundary)
                lossJump = self.gamma_interface*self.loss(torch.matmul(self.U_interface, AB_pred), self.B_interface)
                loss = lossContinuous + lossBoundary + lossJump
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.loss_history.append(loss.item())
            self.lossContinuos_history.append(lossContinuous.item())
            self.lossBoundary_history.append(lossBoundary.item())
            self.lossJump_history.append(lossJump.item())
            if (epoch+1) % 100 == 0:
                print(f'Adam(or SGD) optimizer {epoch}th Loss {loss.item()}')
        # self.optimizer.step(self.loss_func)
    
    def predict(self, grid, x, f_A, f_B, f_F, f_AB):
        # 训练数据集不包括x=0, 测试时输入x=0，得到的pred_AB不能保证pre_u满足边界条件
        self.dnn.eval()
        pred_AB = self.dnn(torch.tensor(grid).float().to(device)).reshape((len(grid),2))
        pred_AB = pred_AB.detach().cpu().numpy()
        pred_u = np.zeros(len(x))
        # pred_u[0] = pred_AB[1, 0]*coeff[0][0]+pred_AB[1, 1]*coeff[0][1]+coeff[0][2]
        A_x, B_x = [], []
        A_x_pred, B_x_pred = [], []
        def f_AB_pred(x):
            result = pred_AB[0]
            for i in range(1, N):
                result = np.where((grid[i-1] < x) & (x <= grid[i]), pred_AB[i], result)
            return result
        for i in range(len(x)):
            pred_u[i] = f_AB_pred(x[i])[0]*f_A(x[i])+f_AB_pred(x[i])[1]*f_B(x[i])+f_F(x[i])
            # pred_u[i] = f_AB(x[i])[0]*f_A(x[i])+f_AB(x[i])[1]*f_B(x[i])+f_F(x[i])
            A_x.append(f_AB(x[i])[0])
            A_x_pred.append(f_AB_pred(x[i])[0])
            B_x.append(f_AB(x[i])[1])
            B_x_pred.append(f_AB_pred(x[i])[1])

        fig = plt.figure(figsize=(7, 3), dpi=150)
        plt.subplot(1, 2, 1)
        plt.plot(x, A_x, label='ground truth')
        plt.plot(x, A_x_pred, label='prediction')
        plt.title('A(x)')

        plt.subplot(1, 2, 2)
        plt.plot(x, B_x, label='ground truth')
        plt.plot(x, B_x_pred, label='prediction')
        plt.title('B(x)')
        plt.legend()
        # plt.show()
        return pred_u


N = 11
q1 = lambda x: 5.0
q2 = lambda x: 0.1*(4+32*x)
F = lambda x: x
layers = [N, 16, 32, 2*N]

q = lambda x: np.where(x<=0.5, q1(x), q2(x))
grid = np.linspace(0, 1, N)
qLeft, qRight = q(grid), q(grid)
qRight[int((N-1)/2)] = q2(grid[int((N-1)/2)])

u1, u2, U, B, f_AB, f_A, f_B, f_F = tfpm(grid, qLeft, qRight, F)
model = PhysicsInformedNN(grid, U, B, layers)
start_time = time.time()
model.train(-1, 400)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time:.6f} seconds")

N_fine = 101
grid_fine = np.linspace(0, 1, N_fine)
prediction = model.predict(grid, grid_fine, f_A, f_B, f_F, f_AB)

q = lambda x: np.where(x<=0.5, q1(x), q2(x))
grid = np.linspace(0, 1, N_fine)
qLeft, qRight = q(grid), q(grid)
qRight[int((N_fine-1)/2)] = q2(grid[int((N_fine-1)/2)])
u1, u2, U, B, f_AB, f_A, f_B, f_F = tfpm(grid, qLeft, qRight, F)
print(f'test error on grid with resolution {N_fine}: {1/N_fine*np.sum(np.abs(prediction-np.concatenate((u1, u2[1:]))))}')
fig = plt.figure(figsize=(7, 3), dpi=150)
plt.subplot(1, 2, 1)
plt.plot(grid_fine[:int(N_fine/2+1)], u1, 'r-', label='Ground Truth', alpha=1., zorder=0)
plt.plot(grid_fine[int(N_fine/2+1):], u2[1:], 'r-', alpha=1., zorder=0)
plt.plot(grid_fine[:int(N_fine/2+1)], prediction[:int(N_fine/2+1)], 'b-', label='prediction', alpha=1., zorder=0)
plt.plot(grid_fine[int(N_fine/2+1):], prediction[int(N_fine/2+1):], 'b-', alpha=1., zorder=0)
plt.legend(loc='best')
plt.xlabel("x")
plt.ylabel("$u$")
plt.grid()

plt.subplot(1, 2, 2)
plt.title("training loss")
plt.plot(model.lossBoundary_history, label='boundary loss')
plt.plot(model.loss_history, label='total loss')
plt.plot(model.lossContinuos_history, label='continuous loss')
plt.plot(model.lossJump_history, label='jump loss')
plt.legend()
plt.yscale("log")
plt.xlabel("epoch")

plt.tight_layout()
plt.show(block=True)
