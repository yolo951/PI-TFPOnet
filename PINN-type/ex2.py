
import torch
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy
from scipy.integrate import quad
from math import sqrt
from scipy import interpolate
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# solve equation -u''(x)+q(x)u(x)=F(x)
def discontinuous_tfpm(f, N):

    def q(x):
        return np.where(x<=0.5, 5000.0, 100.0*(4+32*x))

    def q2(x):
        return 100.0*(4+32*x)

    def F(x):
        return f(x)
        # return interpolate_f(x)[k]

    def integrand_linear(s):  # for -u''=f
        G = np.where(y >= s, s-y, 0)
        return F(s)*G

    def integrandp_linear(s):  # for -u''=f
        G = np.where(y >= s, -1.0, 0)
        return F(s)*G

    def integrand_sinh(s):  # for -u''+bu = f, b>0
        G = 1/2/sqrt(b)*np.where(y >= s, np.sinh(sqrt(b)*(s-y)), np.sinh(sqrt(b)*(y-s)))
        return F(s) * G

    def integrandp_sinh(s):
        G = 1/2*np.where(y > s, -np.cosh(sqrt(b)*(s-y)), np.where(y<s, np.cosh(sqrt(b)*(y-s)), 0.0))
        return F(s) * G

    def integrand_airy(s):  # for -u'' + (ax+b)u = f
        G = np.pi/2*np.where(z >= s, airy(s)[2] * airy(z)[0] - airy(s)[0] * airy(z)[2],
                              airy(s)[0] * airy(z)[2] - airy(s)[2] * airy(z)[0])
        return np.power(np.abs(a), -2/3)*G*F(s/np.cbrt(a)-b/a)

    def integrandp_airy(s):
        G = np.pi/2 * np.where(z > s, airy(s)[2] * airy(z)[1] - airy(s)[0] * airy(z)[3],
                               np.where(z<s, airy(s)[0] * airy(z)[3] - airy(s)[2] * airy(z)[1], 0.0))
        return 1/np.cbrt(a)*G*F(s/np.cbrt(a)-b/a)

    # interpolate_f = interpolate.interp1d(np.linspace(0, 1, N), f)
    x = np.zeros((N), dtype=np.float64)
    x1 = np.zeros((int((N+1)/2)), dtype=np.float64)
    x2 = np.zeros((int((N + 1) / 2)), dtype=np.float64)
    U = np.zeros((2*(N-1), 2*(N-1)), dtype=np.float64)
    B = np.zeros((2*(N-1),), dtype=np.float64)

    grid = np.linspace(0, 1, N)
    cRight = q(grid)
    cLeft = q(grid)
    cRight[int((N-1)/2)] = q2(grid[int((N-1)/2)])
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
            lambda1p = 0.0
            mu1p = 1.0
            lambda1pp = 0.0
            mu1pp = 0.0
        else:
            lambda1 = np.exp(grid[0] * sqrt(b))
            mu1 = np.exp(-grid[0] * sqrt(b))
            y = grid[0]
            F1 = quad(integrand_sinh, grid[0], grid[1])[0]
            lambda1p = sqrt(b) * np.exp(grid[0] * sqrt(b))
            mu1p = -sqrt(b) * np.exp(-grid[0] * sqrt(b))
            lambda1pp = b * np.exp(grid[0] * sqrt(b))
            mu1pp = b * np.exp(-grid[0] * sqrt(b))
    else:
        z1 = q(grid[0]) * np.power(np.abs(a), -2 / 3)  # 如果直接使用np.power(a, -2/3),可能会计算出来复数
        z2 = q(grid[1]) * np.power(np.abs(a), -2 / 3)
        lambda1 = airy(z1)[0]
        mu1 = airy(z1)[2]
        z = z1
        F1 = quad(integrand_airy, z1, z2)[0] if z2 >= z1 else quad(integrand_airy, z2, z1)[0]
        lambda1p = np.cbrt(a)*airy(z1)[1]
        mu1p = np.cbrt(a)*airy(z1)[3]
        lambda1pp = z1 * airy(z1)[0]*np.cbrt(a)**2
        mu1pp = z1 * airy(z1)[2]*np.cbrt(a)**2
    U[0,:2] = np.array([-lambda1, -mu1])
    B[0] = F1
    coeff = [[lambda1, mu1, F1]]
    coeffp = [[lambda1p, mu1p]]
    coeffpp = [[lambda1pp, mu1pp]]
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
                lambda1pp = 0.0
                mu1pp = 0.0
            else:
                # b1 = abs(b1)
                # b2 = abs(b2)
                lambda1 = np.exp(sqrt(b1) * grid[i // 2 + 1])
                gamma1 = sqrt(b1) * lambda1
                mu1 = np.exp(-sqrt(b1) * grid[i // 2 + 1])
                delta1 = -sqrt(b1) * mu1
                y, b = grid[i // 2 + 1], b1
                F1 = quad(integrand_sinh, grid[i // 2], grid[i // 2 + 1])[0]
                G1 = quad(integrandp_sinh, grid[i // 2], grid[i // 2 + 1])[0]
                lambda1pp = b1 * lambda1
                mu1pp = b1 * mu1
        else:
            z1 = cRight[i//2] * np.power(np.abs(a1), -2 / 3)
            z2 = cLeft[i//2+1]* np.power(np.abs(a1), -2 / 3)
            lambda1, gamma1, mu1, delta1 = airy(z2)
            gamma1 *= np.cbrt(a1)
            delta1 *= np.cbrt(a1)
            a, b, z = a1, b1, z2
            F1 = quad(integrand_airy, z1, z2)[0] if z2 >= z1 else quad(integrand_airy, z2, z1)[0]
            G1 = quad(integrandp_airy, z1, z2)[0] if z2 >= z1 else quad(integrandp_airy, z2, z1)[0]
            lambda1pp = z2 * lambda1*np.cbrt(a1)**2
            mu1pp = z2 * mu1*np.cbrt(a1)**2
        if abs(a2) < 1e-8:
            if abs(b2) < 1e-8:
                lambda2, gamma2, delta2 = 1, 0, 1
                mu2 = grid[i//2+1]
                y = grid[i//2 + 1]
                F2 = quad(integrand_linear, grid[i//2+1], grid[i//2+2])[0]
                G2 = 0.0
            else:
                lambda2 = np.exp(sqrt(b2) * grid[i // 2 + 1])
                gamma2 = sqrt(b2) * lambda2
                mu2 = np.exp(-sqrt(b2) * grid[i // 2 + 1])
                delta2 = -sqrt(b2) * mu2
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
        coeffp.append([gamma1, delta1])
        coeffpp.append([lambda1pp, mu1pp])
    c = np.polyfit(grid[-2:], np.array([cRight[-2], cLeft[-1]]), 1)
    a, b = c[0], c[1]
    # a, b = interp_coeff[-1]
    if abs(a)<1e-8:
        if abs(b)<1e-8:
            lambda1 = 1.0
            mu1 = grid[-1]
            y = grid[-1]
            F1 = quad(integrand_linear, grid[-2], grid[-1])[0]
            lambda1p = 0.0
            mu1p = 1.0
            lambda1pp = 0.0
            mu1pp = 0.0
        else:
            # b = abs(b)
            lambda1 = np.exp(sqrt(b)*grid[-1])
            mu1 = np.exp(-sqrt(b)*grid[-1])
            y = grid[-1]
            F1 = quad(integrand_sinh, grid[-2], grid[-1])[0]
            lambda1p = sqrt(b) * np.exp(grid[-1] * sqrt(b))
            mu1p = -sqrt(b) * np.exp(-grid[-1] * sqrt(b))
            lambda1pp = b * np.exp(grid[-1] * sqrt(b))
            mu1pp = b * np.exp(-grid[-1] * sqrt(b))
    else:
        z1 = q(grid[-2]) * np.power(np.abs(a), -2 / 3)
        z2 = q(grid[-1]) * np.power(np.abs(a), -2 / 3)
        lambda1 = airy(z2)[0]
        mu1 = airy(z2)[2]
        z = z2
        F1 = quad(integrand_airy, z1, z2)[0] if z2 >= z1 else quad(integrand_airy, z2, z1)[0]
        lambda1p = airy(z2)[1]*np.cbrt(a)
        mu1p = airy(z2)[3]*np.cbrt(a)
        lambda1pp = z2 * airy(z2)[0]*np.cbrt(a)**2
        mu1pp = z2 * airy(z2)[2]*np.cbrt(a)**2
    U[2*N-3, -2:] = np.array([-lambda1, -mu1])
    B[2*N-3] = F1
    B[N-2] -= 1.0
    B[N-1] -= 1.0
    coeff.append([lambda1, mu1, F1])
    coeffp.append([lambda1p, mu1p])
    coeffpp.append([lambda1pp, mu1pp])

    max_idx = np.argmax(np.abs(U), axis=0)
    M_inverse = np.diag(1/np.abs(U[max_idx,range(U.shape[1])]))
    scaled_U = np.matmul(U, M_inverse)
    AB = np.linalg.solve(scaled_U, B).flatten() # scaled AB

    scale = np.diag(M_inverse)
    scale1 = scale[::2]
    scale2 = scale[1::2]
    for i in range(len(coeffp)):
        j = i-1 if i>0 else 0
        coeff[i][0] *= scale1[j]
        coeff[i][1] *= scale2[j]
        coeffp[i][0] *= scale1[j]
        coeffp[i][1] *= scale2[j]
        coeffpp[i][0] *= scale1[j]
        coeffpp[i][1] *= scale2[j]

    # AB = np.linalg.solve(U, B).flatten()
    each_x = np.zeros(N)
    # boundary val = 0
    # each_x[0] = AB[0]*coeff[0][0]+AB[1]*coeff[0][1]+coeff[0][2]
    for i in range(1, N-1):
        each_x[i] = AB[2*(i-1)]*coeff[i][0]+AB[2*(i-1)+1]*coeff[i][1]+coeff[i][2]
    x[:] = each_x

    x1[:] = x[:int((N+1)/2)]
    x2[:] = x[int((N+1)/2)-1:]
    x2[0] += 1
    return x1, x2, scaled_U, B, coeff, AB, coeffp, coeffpp


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
    def __init__(self, X, U, B, layers, coeff, coeffpp):
        
        self.x = torch.tensor(X.reshape((-1, 1))).float().to(device)
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
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=3e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.6)
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
        self.coeff = torch.tensor(coeff).float().to(device)
        self.coeffpp = torch.tensor(coeffpp).float().to(device)
        
    def loss_equ(self):
        q = lambda x: torch.where(x<=0.5, 5000.0, 100.0*(4+32*x))
        qh = lambda x: torch.where(x<=0.5, 5000.0, 100.0*(4+32*x))
        f = lambda x: x
        AB = self.dnn(self.x)
        u0 = torch.sum(self.coeff[1:-1, :2] * AB[:-1], dim=1)
        upp0 = torch.sum(self.coeffpp[1:-1, :2] * AB[:-1], dim=1)
        # FG = \int F*G, -FG''+qh*FG=F
        # then -FG''+q*FG = -FG''+qh*FG+(q-qh)*FG=F+(q-qh)*FG
        # -u0''+qh*u0=0, loss=-u0''+q*u0=-u0''+qh*u0+(q-qh)*u0=(q-qh)*u0
        loss = self.loss(upp0, q(self.x[1:-1]).flatten()*u0) + \
            self.loss(qh(self.x[1:-1]).flatten()*self.coeff[1:-1, 2], q(self.x[1:-1]).flatten()*self.coeff[1:-1, 2])
        # loss = self.loss(-upp+q(self.x[:-1]).flatten()*u, f(self.x[:-1]).flatten())
        return loss

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
    
    def predict(self, x, coeff):
        # 训练数据集不包括x=0, 测试时输入x=0，得到的pred_AB不能保证pre_u满足边界条件
        x = torch.tensor(x.reshape((-1, 1))).float().to(device)
        self.dnn.eval()
        pred_AB = self.dnn(x)
        pred_AB = pred_AB.detach().cpu().numpy()
        pred_u = np.zeros(x.numel())
        # pred_u[0] = pred_AB[1, 0]*coeff[0][0]+pred_AB[1, 1]*coeff[0][1]+coeff[0][2]
        for i in range(x.numel()):
            pred_u[i] = pred_AB[i, 0]*coeff[i][0]+pred_AB[i, 1]*coeff[i][1]+coeff[i][2]
        return pred_u

# 输入gridx, A0*Ai(0)+B0*Bi(0)+F0=0, A0=A1, B0=B1


N_x = 21
layers = [1, 32, 32, 32, 2]
gridx = np.linspace(0, 1, N_x)
f = lambda x: x
u1, u2, U, B, coeff, AB, coeffp, coeffpp = discontinuous_tfpm(f, N_x)
# plt.plot(gridx[:int(N_x/2+1)], u1, 'r-', label='Ground Truth', alpha=1., zorder=0)
# plt.plot(gridx[int(N_x/2+1):], u2[1:], 'r-', alpha=1., zorder=0)
# plt.show()


model = PhysicsInformedNN(gridx, U, B, layers, coeff, coeffpp)
model.train(-1, 100000)
prediction = model.predict(gridx, coeff)

plt.figure()
# plt.plot(gridx, groundTruth[:], '-', label='Ground Truth', alpha=1.0, zorder=0) #analytical
plt.plot(gridx[:int(N_x/2+1)], prediction[:int(N_x/2+1)], 'b-', label='prediction', alpha=1., zorder=0)
plt.plot(gridx[int(N_x/2+1):], prediction[int(N_x/2+1):], 'b-', alpha=1., zorder=0)
plt.legend(loc='best')
plt.show()

fig = plt.figure(figsize=(7, 3), dpi=150)
plt.subplot(1, 2, 1)
plt.plot(gridx[:int(N_x/2+1)], u1, 'r-', label='Ground Truth', alpha=1., zorder=0)
plt.plot(gridx[int(N_x/2+1):], u2[1:], 'r-', alpha=1., zorder=0)
plt.plot(gridx[:int(N_x/2+1)], prediction[:int(N_x/2+1)], 'b-', label='prediction', alpha=1., zorder=0)
plt.plot(gridx[int(N_x/2+1):], prediction[int(N_x/2+1):], 'b-', alpha=1., zorder=0)
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
