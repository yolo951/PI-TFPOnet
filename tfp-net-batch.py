
import sys
sys.path.insert(0, '../nips/')
import torch
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy
from scipy.integrate import quad
from math import sqrt
from scipy import interpolate
import generate_data_1d

# warnings.filterwarnings('ignore')

# np.random.seed(1234)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')





# solve equation -u''(x)+q(x)u(x)=F(x)
def discontinuous_tfpm(f, N):

    def q(x):
        return np.where(x<=0.5, 5000.0, 100*(4+32*x))

    def q2(x):
        return 100*(4+32*x)

    def F(x):
        return f(x)
        # return interpolate_f(x)[k]

    def integrand_linear(s):  # for -u''=f
        G = np.where(y >= s, s-y, 0)
        return F(s)*G

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
        else:
            # b = abs(b)
            lambda1 = np.exp(grid[0] * sqrt(b))
            mu1 = np.exp(-grid[0] * sqrt(b))
            y = grid[0]
            F1 = quad(integrand_sinh, grid[0], grid[1])[0]
    else:
        z1 = q(grid[0]) * np.power(np.abs(a), -2 / 3)  # 如果直接使用np.power(a, -2/3),可能会计算出来复数
        z2 = q(grid[1]) * np.power(np.abs(a), -2 / 3)
        lambda1 = airy(z1)[0]
        mu1 = airy(z1)[2]
        z = z1
        F1 = quad(integrand_airy, z1, z2)[0] if z2 >= z1 else quad(integrand_airy, z2, z1)[0]
    U[0,:2] = np.array([-lambda1, -mu1])
    B[0] = F1
    coeff = [[lambda1, mu1, F1]]
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
    c = np.polyfit(grid[-2:], np.array([cRight[-2], cLeft[-1]]), 1)
    a, b = c[0], c[1]
    # a, b = interp_coeff[-1]
    if abs(a)<1e-8:
        if abs(b)<1e-8:
            lambda1 = 1.0
            mu1 = grid[-1]
            y = grid[-1]
            F1 = quad(integrand_linear, grid[-2], grid[-1])[0]
        else:
            # b = abs(b)
            lambda1 = np.exp(sqrt(b)*grid[-1])
            mu1 = np.exp(-sqrt(b)*grid[-1])
            y = grid[-1]
            F1 = quad(integrand_sinh, grid[-2], grid[-1])[0]
    else:
        z1 = q(grid[-2]) * np.power(np.abs(a), -2 / 3)
        z2 = q(grid[-1]) * np.power(np.abs(a), -2 / 3)
        lambda1 = airy(z2)[0]
        mu1 = airy(z2)[2]
        z = z2
        F1 = quad(integrand_airy, z1, z2)[0] if z2 >= z1 else quad(integrand_airy, z2, z1)[0]
    U[2*N-3, -2:] = np.array([-lambda1, -mu1])
    B[2*N-3] = F1
    B[N-2] -= 1.0
    B[N-1] -= 1.0
    coeff.append([lambda1, mu1, F1])


    # U_with_inf = np.where(U == 0, np.inf, U)
    min_idx = np.argmax(np.abs(U), axis=0)
    M_inverse = np.diag(1/np.abs(U[min_idx,range(U.shape[1])]))
    scaled_U = np.matmul(U, M_inverse)
    AB = np.linalg.solve(scaled_U, B).flatten() # scaled AB
    # update scaled local basis
    coeff[0][:2] = scaled_U[0, :2]
    for j in range(1, N):
        coeff[j][:2] = scaled_U[2*j-1, 2*(j-1):2*j]
    
    # AB = np.linalg.solve(U, B).flatten()
    each_x = np.zeros(N)
    each_x[0] = AB[0]*coeff[0][0]+AB[1]*coeff[0][1]+coeff[0][2]
    for i in range(1, N):
        each_x[i] = AB[2*(i-1)]*coeff[i][0]+AB[2*(i-1)+1]*coeff[i][1]+coeff[i][2]
    x[:] = each_x

    x1[:] = x[:int((N+1)/2)]
    x2[:] = x[int((N+1)/2)-1:]
    x2[0] += 1
    return x1, x2, scaled_U, B, coeff, AB


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
    def __init__(self, interior_x, interior_U, interior_B, boundary_x, boundary_U, boundary_B, layers):
        
        self.interior_x = torch.tensor(interior_x, requires_grad=True).unsqueeze(dim=-1).float().to(device)
        self.interior_U = torch.tensor(interior_U).float().to(device)
        self.interior_B = torch.tensor(interior_B).float().to(device)
        self.boundary_x = torch.tensor(boundary_x, requires_grad=True).unsqueeze(dim=-1).float().to(device)
        self.boundary_U = torch.tensor(boundary_U).float().to(device)
        self.boundary_B = torch.tensor(boundary_B).float().to(device)

        self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.interior_x,
                                                                                       self.interior_U, self.interior_B),
                                                                                       batch_size=256, shuffle=True)
        # self.x = torch.tensor(X.reshape((-1, 1)), requires_grad=True).float().to(device)
        # self.U = torch.tensor(U).float().to(device)
        # self.B = torch.tensor(B).float().to(device)

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
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_Adam, step_size=200, gamma=0.6)
        self.iter = 0
        self.loss = torch.nn.MSELoss()
        self.loss_history = []


    def loss_func(self):
        # boundary condition
        epoch_loss = 0
        # boundary condition
        self.optimizer_Adam.zero_grad()  
        AB_pred = self.dnn(self.boundary_x)  
        loss_boundary = self.loss(torch.einsum('ik,ik->i', AB_pred, self.boundary_U), self.boundary_B)  
        loss_boundary.backward()
        self.optimizer_Adam.step()
        epoch_loss += loss_boundary.item()
        for x, U, B in self.train_loader:
            self.optimizer_Adam.zero_grad()
            AB_pred = self.dnn(x).reshape((-1, 4))  
            loss = self.loss(torch.einsum('ik,ijk->ij', AB_pred, U), B)  
            loss.backward()
            self.optimizer_Adam.step()
            epoch_loss += loss.item()

        
        self.iter += 1
        self.scheduler.step()
        self.loss_history.append(epoch_loss)
        if (self.iter+1) % 1 == 0:
            print(f'LBFGS optimizer {self.iter}th Loss {loss.item()}')
        return loss
    
    def train(self, nIter):
        self.dnn.train()
        for epoch in range(nIter):
            epoch_loss = 0
            # boundary condition
            self.optimizer_Adam.zero_grad()  
            AB_pred = self.dnn(self.boundary_x)  
            loss_boundary = self.loss(torch.einsum('ik,ik->i', AB_pred, self.boundary_U), self.boundary_B)  
            loss_boundary.backward()
            self.optimizer_Adam.step()
            epoch_loss += loss_boundary.item()

            for x, U, B in self.train_loader:
                self.optimizer_Adam.zero_grad()
                AB_pred = self.dnn(x).reshape((-1, 4))  
                loss = self.loss(torch.einsum('ik,ijk->ij', AB_pred, U), B)  
                loss.backward()
                self.optimizer_Adam.step()
                epoch_loss += loss.item()

            self.scheduler.step()
            self.loss_history.append(epoch_loss)
            if (epoch+1) % 100 == 0:
                print(f'Adam optimizer {epoch}th Loss {loss.item()}')
                
        # Backward and optimize
        self.optimizer.step(self.loss_func)
    
    def predict(self, x, coeff):
        x = torch.tensor(x.reshape((-1, 1))).float().to(device)
        self.dnn.eval()
        pred_AB = self.dnn(x)
        pred_AB = pred_AB.detach().cpu().numpy()
        pred_u = np.zeros(x.numel())
        for i in range(1, x.numel()):
            pred_u[i] = pred_AB[i, 0]*coeff[i][0]+pred_AB[i, 1]*coeff[i][1]+coeff[i][2]
        return pred_u




N_x = 1001
layers = [1, 20, 20, 20, 20, 20, 20, 2]
gridx = np.linspace(0, 1, N_x)
f = lambda x: x
u1, u2, U, B, coeff, AB = discontinuous_tfpm(f, N_x)
# plt.plot(gridx[:501], u1.flatten(), label='u1')
# plt.plot(gridx[500:], u2.flatten(), label='u2')
# plt.legend()
# plt.show()


reshaped_U = np.zeros((U.shape[0], U.shape[1]+2))
reshaped_U[0, :2] = U[0, :2]
reshaped_U[1:, 2:] = U[1:, :]

sparse_U = np.zeros((U.shape[0], 4))
sparse_U[0, :2] = reshaped_U[0, :2]
for j in range(1, N_x-1):
    sparse_U[2*j-1] = reshaped_U[2*j-1, 2*j:2*j+4]
    sparse_U[2*j] = reshaped_U[2*j, 2*j:2*j+4]
sparse_U[2*(N_x-1)-1, :2] = reshaped_U[2*(N_x-1)-1, -2:]
interior_U = sparse_U[1:-1].reshape((N_x-2, 2, 4))
boundary_U = np.stack((sparse_U[0, :2], sparse_U[-1, :2]))
interior_B = B[1:-1].reshape((N_x-2, 2))
boundary_B = np.array([B[0], B[-1]])
interior_x = np.column_stack((gridx[:-2], gridx[1:-1]))
boundary_x = np.array([gridx[0], gridx[-1]])


model = PhysicsInformedNN(interior_x, interior_U, interior_B, boundary_x, boundary_U, boundary_B, layers)
model.train(2000)

prediction = model.predict(gridx, coeff)  #evaluate model




# plt.figure()
# # plt.plot(gridx, groundTruth[:], '-', label='Ground Truth', alpha=1.0, zorder=0) #analytical
# plt.plot(gridx, prediction, 'k-', label='prediction', alpha=1., zorder=0) #PINN
# plt.legend(loc='best')
# plt.show()

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
plt.plot(model.loss_history)
plt.yscale("log")
plt.xlabel("epoch")

plt.tight_layout()
plt.show(block=True)
