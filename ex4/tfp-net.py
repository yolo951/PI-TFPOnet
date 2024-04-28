# import sys
# sys.path.insert(0, '../Utilities/')

import torch
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
# from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import warnings

# warnings.filterwarnings('ignore')

np.random.seed(1234)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# import sys
# sys.path.append('D:\pycharm\pycharm_project\TFPONet')
import numpy as np
from scipy.special import airy
from scipy.integrate import quad
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, cos, e
from scipy import interpolate



# solve equation -u''(x)+q(x)u(x)=F(x)
def discontinuous_tfpm(N, eps):

    def func_a(x):
        return np.where(x <= 0.25, 1 + x,
                                  np.where(x <= 0.5, x, np.ones_like(x)))
    def func_b(x):
        return 1 / eps * np.where(x <= 0.25, 1 + np.exp(x),
                                  np.where(x <= 0.5, 1 - np.log(x), 4 * np.ones_like(x)))
    def q(x):
        return func_a(x) * func_b(x)
    def q2(x):
        return 1 / eps * (1 - np.log(x))*x
    def q3(x):
        return 1 / eps * 4 * np.ones_like(x)


    def gety(x):
        if x<=0.25:
            return quad(integrand_y, 0, x)[0]
        elif 0.25<x<=0.5:
            return quad(integrand_y, 0, 0.25)[0]+quad(integrand_y, 0.25, x)[0]
        else:
            return quad(integrand_y, 0, 0.25)[0]+quad(integrand_y, 0.25, 0.5)[0]+quad(integrand_y, 0.5, x)[0]
    integrand_y = lambda x: 1 / func_a(x)
    gridx = np.linspace(0, 1, N)
    val_a = func_a(gridx)
    grid = np.array([gety(gridx[i]) for i in range(N)])



    def F(x):
        # return np.zeros_like(x)
        return 1/eps*np.ones_like(x)

    def integrand_linear(s):  # for -u''=f
        G = np.where(y >= s, s-y, 0)
        return F(s)*G

    def integrand_sin(s):  # for -u''+bu = f, b<0
        G = 1/sin(1)*np.where(y >= s, np.sin(1-y)*sin(s), np.sin(1-s)*sin(y))
        return F(s) * G

    def integrandp_sin(s):
        G = 1/sin(1)*np.where(y >= s, -np.cos(1-y)*sin(s), np.sin(1-s)*cos(y))
        return F(s) * G

    def integrand_sinh(s):  # for -u''+bu = f, b>0
        G = 1/2/sqrt(b)*np.where(y >= s, np.sinh(sqrt(b)*(s-y)), np.sinh(sqrt(b)*(y-s)))
        return F(s) * G

    def integrandp_sinh(s):
        G = 1/2*np.where(y >= s, -np.cosh(sqrt(b)*(s-y)), np.cosh(sqrt(b)*(y-s)))
        return F(s) * G

    def integrand_airy(s):  # for -u'' + (ax+b)u = f
        G = np.pi/2*np.where(z >= s, airy(s)[2] * airy(z)[0] - airy(s)[0] * airy(z)[2],
                              airy(s)[0] * airy(z)[2] - airy(s)[2] * airy(z)[0])
        return np.power(np.abs(a), -2/3)*G*F(s/np.cbrt(a)-b/a)

    def integrandp_airy(s):
        G = np.pi/2 * np.where(z >= s, airy(s)[2] * airy(z)[1] - airy(s)[0] * airy(z)[3],
                              airy(s)[0] * airy(z)[3] - airy(s)[2] * airy(z)[1])
        return 1/np.cbrt(a)*G*F(s/np.cbrt(a)-b/a)

    x = np.zeros((N, ), dtype=np.float64)
    U = np.zeros((2*(N-1), 2*(N-1)), dtype=np.float64)
    B = np.zeros((2*(N-1),), dtype=np.float64)
    cRight = q(gridx)
    cLeft = q(gridx)
    cRight[int((N - 1) / 4)] = q2(gridx[int((N - 1) / 4)])
    cRight[int((N - 1) / 2)] = q3(gridx[int((N - 1) / 2)])
    # interp_coeff = get_coeff(grid)
    c = np.polyfit(grid[:2], np.array([cRight[0], cLeft[1]]), 1)
    a, b = c[0], c[1]
    # a, b = interp_coeff[0]
    if abs(a) < 1e-8:
        if abs(b) < 1e-8:
            lambda1 = 1.0
            mu1 = grid[0]
            y = grid[0]
            F1 = quad(integrand_linear, grid[0], grid[1])[0]
        elif b > 0:
            # b = abs(b)
            lambda1 = np.exp(grid[0] * sqrt(b))
            mu1 = np.exp(-grid[0] * sqrt(b))
            y = grid[0]
            F1 = quad(integrand_sinh, grid[0], grid[1])[0]
        else:
            b = abs(b)
            lambda1 = np.sin(grid[0] * sqrt(b))
            mu1 = np.cos(grid[0] * sqrt(b))
            y = grid[0]
            F1 = quad(integrand_sin, grid[0], grid[1])[0]
    else:
        z1 = q(gridx[0]) * np.power(np.abs(a), -2 / 3)  # 如果直接使用np.power(a, -2/3),可能会计算出来复数
        z2 = q(gridx[1]) * np.power(np.abs(a), -2 / 3)
        lambda1 = airy(z1)[0]
        mu1 = airy(z1)[2]
        z = z1
        F1 = quad(integrand_airy, z1, z2)[0] if z2 >= z1 else quad(integrand_airy, z2, z1)[0]
    U[0,:2] = np.array([-lambda1, -mu1])
    B[0] = F1
    coeff = []
    for i in range(1, 2*N-3, 2):
        c= np.polyfit(grid[i//2:i//2+2], np.array([cRight[i//2], cLeft[i//2+1]]), 1)
        a1, b1 = c[0], c[1]
        c = np.polyfit(grid[i//2+1:i//2+3], np.array([cRight[i//2+1], cLeft[i//2+2]]), 1)
        a2, b2 = c[0], c[1]
        # a1, b1 = interp_coeff[i // 2]
        # a2, b2 = interp_coeff[i // 2 + 1]
        if abs(a1) < 1e-8:
            if abs(b1) < 1e-8:
                lambda1, gamma1, delta1 = 1.0, 0.0, 1.0
                mu1 = grid[i//2+1]
                y = grid[i//2 + 1]
                F1 = quad(integrand_linear, grid[i//2], grid[i//2+1])[0]
                G1 = -quad(F, grid[i//2], grid[i//2+1])[0]
            elif b1>0:
                # b1 = abs(b1)
                # b2 = abs(b2)
                lambda1 = np.exp(sqrt(b1) * grid[i // 2 + 1])
                gamma1 = sqrt(b1) * lambda1
                mu1 = np.exp(-sqrt(b1) * grid[i // 2 + 1])
                delta1 = -sqrt(b1) * mu1
                y, b = grid[i // 2 + 1], b1
                F1 = quad(integrand_sinh, grid[i // 2], grid[i // 2 + 1])[0]
                G1 = quad(integrandp_sinh, grid[i // 2], grid[i // 2 + 1])[0]
            else:
                b1 = abs(b1)
                lambda1 = np.sin(sqrt(b1) * grid[i // 2 + 1])
                gamma1 = sqrt(b1) * np.cos(sqrt(b1) * grid[i // 2 + 1])
                mu1 = np.cos(sqrt(b1) * grid[i // 2 + 1])
                delta1 = -sqrt(b1) * np.sin(sqrt(b1) * grid[i // 2 + 1])
                y, b = grid[i // 2 + 1], b1
                F1 = quad(integrand_sin, grid[i // 2], grid[i // 2 + 1])[0]
                G1 = quad(integrandp_sin, grid[i // 2], grid[i // 2 + 1])[0]
        else:
            z1 = cRight[i//2] * np.power(np.abs(a1), -2 / 3)
            z2 = cLeft[i//2+1]* np.power(np.abs(a1), -2 / 3)
            lambda1, gamma1, mu1, delta1 = airy(z2)
            gamma1 *= np.cbrt(a1)
            delta1 *= np.cbrt(a1)
            a, b, z = a1, b1, z2
            F1 = quad(integrand_airy, z1, z2)[0] if z2 >= z1 else quad(integrand_airy, z2, z1)[0]
            G1 = quad(integrandp_airy, z1, z2)[0] if z2 >= z1 else quad(integrandp_airy, z2, z1)[0]
        if abs(a2) < 1e-8:
            if abs(b2) < 1e-8:
                lambda2, gamma2, delta2 = 1.0, 0.0, 1.0
                mu2 = grid[i//2+1]
                y = grid[i//2 + 1]
                F2 = quad(integrand_linear, grid[i//2+1], grid[i//2+2])[0]
                G2 = 0
            elif b2>0:
                lambda2 = np.exp(sqrt(b2) * grid[i // 2 + 1])
                gamma2 = sqrt(b2) * lambda2
                mu2 = np.exp(-sqrt(b2) * grid[i // 2 + 1])
                delta2 = -sqrt(b2) * mu2
                y, b = grid[i // 2 + 1], b2
                F2 = quad(integrand_sinh, grid[i // 2 + 1], grid[i // 2 + 2])[0]
                G2 = quad(integrandp_sinh, grid[i // 2 + 1], grid[i // 2 + 2])[0]
            else:
                b2 = abs(b2)
                lambda2 = np.sin(sqrt(b2) * grid[i // 2 + 1])
                gamma2 = sqrt(b2) * np.cos(sqrt(b2) * grid[i // 2 + 1])
                mu2 = np.cos(sqrt(b2) * grid[i // 2 + 1])
                delta2 = -sqrt(b2) * np.sin(sqrt(b2) * grid[i // 2 + 1])
                y, b = grid[i // 2 + 1], b2
                F2 = quad(integrand_sinh, grid[i // 2 + 1], grid[i // 2 + 2])[0]
                G2 = quad(integrandp_sinh, grid[i // 2 + 1], grid[i // 2 + 2])[0]
        else:
            z3 = cRight[i//2+1] * np.power(np.abs(a2), -2 / 3)
            z4 = cLeft[i//2+2] * np.power(np.abs(a2), -2 / 3)
            lambda2, gamma2, mu2, delta2 = airy(z3)
            gamma2 *= np.cbrt(a2)
            delta2 *= np.cbrt(a2)
            a, b, z = a2, b2, z3
            F2 = quad(integrand_airy, z3, z4)[0] if z4 >= z3 else quad(integrand_airy, z4, z3)[0]
            G2 = quad(integrandp_airy, z3, z4)[0] if z4 >= z3 else quad(integrandp_airy, z4, z3)[0]
        B[i] = F2 - F1
        B[i+1] = G2 - G1
        U[i, i-1:i+3] = np.array([lambda1, mu1, -lambda2, -mu2])
        U[i+1, i - 1:i + 3] = np.array([gamma1, delta1, -gamma2, -delta2])
        coeff.append([lambda1, mu1, F1])
    c = np.polyfit(grid[-2:], np.array([cRight[-2], cLeft[-1]]), 1)
    a, b = c[0], c[1]
    # a, b = interp_coeff[-1]
    if abs(a)<1e-8:
        if abs(b)<1e-8:
            lambda1 = 1.0
            mu1 = grid[-1]
            y = grid[-1]
            F1 = quad(integrand_linear, grid[-2], grid[-1])[0]
        elif b>0:
            # b = abs(b)
            lambda1 = np.exp(sqrt(b)*grid[-1])
            mu1 = np.exp(-sqrt(b)*grid[-1])
            y = grid[-1]
            F1 = quad(integrand_sinh, grid[-2], grid[-1])[0]
        else:
            b = abs(b)
            lambda1 = np.sin(sqrt(b) * grid[-1])
            mu1 = np.cos(sqrt(b) * grid[-1])
            y = grid[-1]
            F1 = quad(integrand_sin, grid[-2], grid[-1])[0]
    else:
        z1 = q(gridx[-2]) * np.power(np.abs(a), -2 / 3)
        z2 = q(gridx[-1]) * np.power(np.abs(a), -2 / 3)
        lambda1 = airy(z2)[0]
        mu1 = airy(z2)[2]
        z = z2
        F1 = quad(integrand_airy, z1, z2)[0] if z2 >= z1 else quad(integrand_airy, z2, z1)[0]
    U[2*N-3, -2:] = np.array([-lambda1, -mu1])
    B[2*N-3] = F1
    B[int((N-1)/2)-1] -= -1.0
    B[int((N-1)/2)] -= 1.0
    B[N-2] -= 1.0
    B[N-1] -= 0.0
    U_with_inf = np.where(U == 0, np.inf, U)
    min_idx = np.argmin(np.abs(U_with_inf), axis=0)
    M_inverse = np.diag(1/np.abs(U[min_idx,range(U.shape[1])]))
    AB = np.matmul(M_inverse, np.linalg.solve(np.matmul(U, M_inverse), B).flatten())
    AB = np.linalg.solve(U, B).flatten()
    each_x = np.zeros(N)
    for i in range(1, N-1):
        each_x[i] = AB[2*(i-1)]*coeff[i-1][0]+AB[2*(i-1)+1]*coeff[i-1][1]+coeff[i-1][2]
    x[:] = each_x
    return x, U, B, coeff





# class DNN(torch.nn.Module):
#     def __init__(self, layers):
#         super(DNN, self).__init__()
        
#         self.depth = len(layers) - 1
#         self.activation = torch.nn.ReLU
        
#         layer_list = list()
#         for i in range(self.depth - 1): 
#             layer_list.append(
#                 ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
#             )
#             layer_list.append(('activation_%d' % i, self.activation()))
            
#         layer_list.append(
#             ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
#         )
#         layerDict = OrderedDict(layer_list)
#         self.layers = torch.nn.Sequential(layerDict)
        
#     def forward(self, x):
#         out = self.layers(x)
#         return out


# class PhysicsInformedNN():
#     def __init__(self, X, U, B, layers):
        
#         self.x = torch.tensor(X, requires_grad=True).float().to(device)
#         self.U = torch.tensor(U).float().to(device)
#         self.B = torch.tensor(B).float().to(device)

#         self.dnn = DNN(layers).to(device)
#         self.optimizer = torch.optim.LBFGS(
#             self.dnn.parameters(), 
#             lr=1.0, 
#             max_iter=50000, 
#             max_eval=50000, 
#             history_size=50,
#             tolerance_grad=1e-5, 
#             tolerance_change=1.0 * np.finfo(float).eps,
#             line_search_fn="strong_wolfe"
#         )
#         self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())
#         self.iter = 0

#     def loss_func(self):
#         AB_pred = self.dnn(self.x)
        
#         loss = torch.norm(torch.matmul(self.U, AB_pred)-self.B)
#         self.optimizer.zero_grad()
#         loss.backward()
        
#         self.iter += 1
#         if self.iter % 1 == 0:
#             print('Loss: {}'.format(loss.item()))
#         return loss
    
#     def train(self, nIter):
#         self.dnn.train()
#         for epoch in range(nIter):
#             AB_pred = self.dnn(self.x)
#             loss = torch.norm(torch.matmul(self.U, AB_pred)-self.B)
            
#             # Backward and optimize
#             self.optimizer_Adam.zero_grad()
#             loss.backward()
#             self.optimizer_Adam.step()
            
#             if epoch % 100 == 0:
#                 print('Loss: {}'.format(loss.item()))
                
#         # Backward and optimize
#         self.optimizer.step(self.loss_func)
    
#     def predict(self, X):
#         x = torch.tensor(X, requires_grad=True).float().to(device)
#         self.dnn.eval()
#         pred = self.dnn(x)
#         pred = pred.detach().cpu().numpy()
#         return pred


# N_x = 2000
# gridx = np.linspace(0, 1, N_x)
# groundTruth, U, B, coeff = discontinuous_tfpm(N=N_x, eps=0.001)
# layers = [N_x, 100, 100, 100, U.shape[0]]

# model = PhysicsInformedNN(gridx, U, B, layers)
# model.train(100)

# AB = model.predict(gridx)  #evaluate model
# prediction = np.zeros(N_x)
# for i in range(1, N_x-1):
#     prediction[i] = AB[2*(i-1)]*coeff[i-1][0]+AB[2*(i-1)+1]*coeff[i-1][1]+coeff[i-1][2]

# # C_Result_original = results[0]
# # C_analytical = (np.exp(x/0.01)-1)/(np.exp(1/0.01)-1)-x
# #### Plot ########
# plt.figure()
# plt.plot(gridx, groundTruth[:], '-', label='Ground Truth', alpha=1.0, zorder=0) #analytical
# plt.plot(gridx, prediction, 'k-', label='prediction', alpha=1., zorder=0) #PINN
# plt.legend(loc='best')
# plt.show()


