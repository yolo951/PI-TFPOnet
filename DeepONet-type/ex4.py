# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:15:49 2024

@author: Ye Li
"""
import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
from scipy import interpolate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(samples=10, begin=0, end=1, random_dim=101, out_dim=101, length_scale=1, interp="cubic", A=0):
    space = GRF(begin, end, length_scale=length_scale, N=random_dim, interp=interp)
    features = space.random(samples, A)
    features = features.reshape((features.shape[0], random_dim, random_dim))
    x_grid = np.linspace(begin, end, out_dim)
    x_data = space.eval_u(features, x_grid, x_grid)
    return x_data  # X_data.shape=(samples,out_dim,out_dim)，每一行表示一个GRF在meshgrid上的取值，共有samples个GRF


class GRF(object):
    def __init__(self, begin=0, end=1, length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(begin, end, num=N)
        x, y = np.meshgrid(self.x, self.x)
        self.z = np.stack((x.flatten(), y.flatten()), axis=-1)
        self.K = np.exp(-0.5*self.distance_matrix(self.z, length_scale))
        self.L = np.linalg.cholesky(self.K + 1e-12 * np.eye(self.N**2))

    def distance_matrix(self, x, length_scale):
        diff = x[:, None] - x[None, :]
        squared_diff = np.sum(diff**2, axis=2)
        grid = squared_diff / length_scale**2
        return grid

    def random(self, n, A):
        u = np.random.randn(self.N**2, n)
        return np.dot(self.L, u).T + A

    def eval_u(self, ys, x, y):
        res = np.zeros((ys.shape[0], x.shape[0],x.shape[0]))
        if self.interp == 'linear':
            order = 1
        elif self.interp == 'cubic':
            order = 3
        for i in range(ys.shape[0]):
            res[i] = interpolate.RectBivariateSpline(self.x, self.x, ys[i], kx=order, ky=order)(
                x,y)
        return res


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
    
    def input_encoding(self, x):
        w = 2.0 * torch.pi / 1
        return torch.hstack([torch.tensor(0*x + 1.0).to(device), torch.cos(1 * w * x), torch.cos(2 * w * x), torch.cos(4 * w * x),  torch.sin(1 * w * x), torch.sin(2 * w * x), torch.sin(4 * w * x)])
    
    def forward(self, x):
        x = self.input_encoding(x)
        out = self.layers(x)
        return out



def c(x,y):
    if x < 1/2:
        a = 16
    else:
        a = 1
    return a 

def b(x,y):
    if x > 1/2:
        a = 2*(1-x)
    else:
        a = 0
    return a
    
def tfpm2d(N,f): 
    h = 1/(2*N)
    U = np.zeros((4*N**2,4*N**2))
    B = np.zeros(4*N**2)
    interpolate_f_2d = interpolate.RegularGridInterpolator((np.linspace(0, 1, f.shape[-1]),np.linspace(0, 1, f.shape[-1])), f)
    F = lambda x, y : interpolate_f_2d((x,y))
    
    #boundary y=0.
    for i in range(0,N):
        x0 = (2*i+1)*h
        y0 = h
        f0 = F(x0,y0).item()
        c0 = c(x0,y0)
        mu0 = np.sqrt(c0)/eps
        xi = (2*i+1)*h
        U[i,[4*i,4*i+1,4*i+2,4*i+3]] = np.array([np.exp(-mu0*h),np.exp(-mu0*h),np.exp(-2*mu0*h),1])
        B[i] = (- f0/c0 + b(xi,0))*np.exp(-mu0*h)
    #boundary y=1.
    for i in range(0,N):
        x0 = (2*i+1)*h
        y0 = 1-h
        f0 = F(x0,y0)
        c0 = c(x0,y0)
        mu0 = np.sqrt(c0)/eps
        xi = (2*i+1)*h
        U[N+i,[4*N*(N-1)+4*i,4*N*(N-1)+4*i+1,4*N*(N-1)+4*i+2,4*N*(N-1)+4*i+3]] = np.array([np.exp(-mu0*h),np.exp(-mu0*h),1,np.exp(-2*mu0*h)])
        B[N+i] = (- f0/c0 + b(xi,1))*np.exp(-mu0*h)
    #boundary x=0.
    for i in range(0,N):
        x0 = h
        y0 = (2*i+1)*h
        f0 = F(x0,y0)
        c0 = c(x0,y0)
        mu0 = np.sqrt(c0)/eps
        yi = (2*i+1)*h
        U[2*N+i,[4*N*i,4*N*i+1,4*N*i+2,4*N*i+3]] = np.array([np.exp(-2*mu0*h),1,np.exp(-mu0*h),np.exp(-mu0*h)])
        B[2*N+i] = (- f0/c0 + b(0,yi))*np.exp(-mu0*h)
    #boundary x=1.
    for i in range(0,N):
        x0 = 1-h
        y0 = (2*i+1)*h
        f0 = F(x0,y0)
        c0 = c(x0,y0)
        mu0 = np.sqrt(c0)/eps
        yi = (2*i+1)*h
        U[3*N+i,[4*N*(i+1)-4,4*N*(i+1)-3,4*N*(i+1)-2,4*N*(i+1)-1]] = np.array([1,np.exp(-2*mu0*h),np.exp(-mu0*h),np.exp(-mu0*h)])
        B[3*N+i] = (- f0/c0 + b(1,yi))*np.exp(-mu0*h)
    
    for i in range(0,N-1):
        for j in range(0,N):
            x0 = (2*i+1)*h
            y0 = (2*j+1)*h
            f0 = F(x0,y0)
            c0 = c(x0,y0)
            mu0 = np.sqrt(c0)/eps
            x1 = (2*i+1)*h + 2*h
            f1 = F(x1,y0)
            c1 = c(x1,y0)
            mu1 = np.sqrt(c1)/eps
            mu = max(mu0,mu1)
            U[4*N+N*i+j,[4*N*j+4*i,4*N*j+4*i+1,4*N*j+4*i+2,4*N*j+4*i+3,4*N*j+4*i+4,4*N*j+4*i+5,4*N*j+4*i+6,4*N*j+4*i+7]] = np.array([np.exp((mu0-mu)*h),np.exp(-(mu0+mu)*h),np.exp(-mu*h),np.exp(-mu*h),-np.exp(-(mu1+mu)*h),-np.exp((mu1-mu)*h),-np.exp(-mu*h),-np.exp(-mu*h)])
            B[4*N+N*i+j] = (f1/c1 - f0/c0)*np.exp(-mu*h)
            U[4*N+N*(N-1)+N*i+j,[4*N*j+4*i,4*N*j+4*i+1,4*N*j+4*i+2,4*N*j+4*i+3,4*N*j+4*i+4,4*N*j+4*i+5,4*N*j+4*i+6,4*N*j+4*i+7]] = np.array([mu0*np.exp((mu0-mu)*h),-mu0*np.exp(-(mu0+mu)*h),0,0,-mu1*np.exp(-(mu1+mu)*h),mu1*np.exp((mu1-mu)*h),0,0])
            B[4*N+N*(N-1)+N*i+j] = 0
            #interface
            if i==int(N/2)-1: 
                B[4*N+N*i+j] = B[4*N+N*i+j] - alpha*np.exp(-mu*h)
                B[4*N+N*(N-1)+N*i+j] = B[4*N+N*(N-1)+N*i+j] - beta*np.exp(-mu*h)
    
    for i in range(0,N):
        for j in range(0,N-1):
            x0 = (2*i+1)*h
            y0 = (2*j+1)*h
            f0 = F(x0,y0)
            c0 = c(x0,y0)
            mu0 = np.sqrt(c0)/eps
            y1 = (2*j+1)*h + 2*h
            f1 = F(x0,y1)
            c1 = c(x0,y1)
            mu1 = np.sqrt(c1)/eps
            mu = max(mu0,mu1)
            U[4*N+2*N*(N-1)+N*j+i,[4*N*j+4*i,4*N*j+4*i+1,4*N*j+4*i+2,4*N*j+4*i+3,4*N*(j+1)+4*i,4*N*(j+1)+4*i+1,4*N*(j+1)+4*i+2,4*N*(j+1)+4*i+3]] = np.array([np.exp(-mu*h),np.exp(-mu*h),np.exp((mu0-mu)*h),np.exp(-(mu0+mu)*h),-np.exp(-mu*h),-np.exp(-mu*h),-np.exp(-(mu1+mu)*h),-np.exp((mu1-mu)*h)])
            B[4*N+2*N*(N-1)+N*j+i] = (f1/c1 - f0/c0)*np.exp(-mu*h)
            U[4*N+3*N*(N-1)+N*j+i,[4*N*j+4*i,4*N*j+4*i+1,4*N*j+4*i+2,4*N*j+4*i+3,4*N*(j+1)+4*i,4*N*(j+1)+4*i+1,4*N*(j+1)+4*i+2,4*N*(j+1)+4*i+3]] = np.array([0,0,mu0*np.exp((mu0-mu)*h),-mu0*np.exp(-(mu0+mu)*h),0,0,-mu1*np.exp(-(mu1+mu)*h),mu1*np.exp((mu1-mu)*h)])
            B[4*N+3*N*(N-1)+N*j+i] = 0
    
    #计算解
    C = np.linalg.solve(U,B)
    up = np.zeros((N,N)) #每个网格中心点值，不包含边界
    for i in range(0,N):
        for j in range(0,N):
            x0 = (2*i+1)*h
            y0 = (2*j+1)*h
            f0 = F(x0,y0)
            c0 = c(x0,y0)
            mu0 = np.sqrt(c0)/eps
            c1 = C[4*N*j+4*i]
            c2 = C[4*N*j+4*i+1]
            c3 = C[4*N*j+4*i+2]
            c4 = C[4*N*j+4*i+3]
            up[j,i] = f0/c0 + c1 + c2 + c3 + c4
    return U, B, C, up

N = 10
ntrain = 1000  
ntest = 200
ntotal = ntrain + ntest
alpha = 1 #interface jump
beta = 0
eps = 1.001
f = generate(samples = ntotal, out_dim=N, length_scale=1)
f = 1. + 0.1*f #训练和测试的f差别小一点

epochs = 20000
learning_rate = 0.001
batch_size = 10
step_size = 2000
gamma = 0.5
model = DNN([7*N**2,512,128,128,512,4*N**2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

k = 0 
x = np.linspace(1/(2*N),1-1/(2*N),N)
xx,yy = np.meshgrid(x,x)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xx, yy, f[k], cmap='rainbow')
ax.title.set_text('generated f(x,y)')
plt.show()

U_total = np.zeros((ntotal,4*N**2,4*N**2))
B_total = np.zeros((ntotal,4*N**2))
C_total = np.zeros((ntotal,4*N**2))
up_total = np.zeros((ntotal,N,N))
f_total = np.zeros((ntotal,N**2))
for k in range(ntotal):              #todo并行加速for large N.
    U, B, C, up = tfpm2d(N,f[k])
    U_total[k] = U
    B_total[k] = B
    C_total[k] = C
    up_total[k] = up
    f_total[k] = f[k].reshape(-1)

f_total = torch.Tensor(f_total).to(device)
U_total = torch.Tensor(U_total).to(device)
B_total = torch.Tensor(B_total).to(device)
C_total = torch.Tensor(C_total).to(device)
f_train = f_total[0:ntrain]
U_train = U_total[0:ntrain]
B_train = B_total[0:ntrain]
C_train = C_total[0:ntrain]
up_train = up_total[0:ntrain]
f_test = f_total[ntrain:ntotal] 
up_test = up_total[ntrain:ntotal] 
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_train,U_train,B_train,C_train), batch_size=batch_size, shuffle=True)
mseloss = torch.nn.MSELoss()


index_jump = []
index_continuous = []
index_boundary = []
for i in range(4*N):
    index_boundary.append(i)
for i in range(0,N-1):
    for j in range(0,N):
        if i==int(N/2)-1: 
            index_jump.append(4*N+N*i+j)
            index_jump.append(4*N+N*(N-1)+N*i+j)
        else:
            index_continuous.append(4*N+N*i+j)
            index_continuous.append(4*N+N*(N-1)+N*i+j)
for i in range(0,N):
    for j in range(0,N-1):
        index_continuous.append(4*N+2*N*(N-1)+N*j+i)
        index_continuous.append(4*N+3*N*(N-1)+N*j+i)
index_jump = torch.LongTensor(index_jump).to(device)
index_continuous = torch.LongTensor(index_continuous).to(device)
index_boundary = torch.LongTensor(index_boundary).to(device)
for i in range(epochs):
    model.train()
    train_mse = 0
    for fb, Ub, Bb, Cb in train_loader:
        optimizer.zero_grad()
        Cb_pred = model(fb)
        U_jump = torch.index_select(Ub, 1, index_jump)
        B_jump = torch.index_select(Bb, 1, index_jump)
        U_continuous = torch.index_select(Ub, 1, index_continuous)
        B_continuous = torch.index_select(Bb, 1, index_continuous)
        U_boundary = torch.index_select(Ub, 1, index_boundary)
        B_boundary = torch.index_select(Bb, 1, index_boundary)
        loss_jump = mseloss(torch.einsum('bri, bi->br', U_jump, Cb_pred), B_jump)
        loss_continuous = mseloss(torch.einsum('bri, bi->br', U_continuous, Cb_pred), B_continuous)
        loss_boundary = mseloss(torch.einsum('bri, bi->br', U_boundary, Cb_pred), B_boundary)
        loss = 1*loss_jump + 100*loss_continuous +1000*loss_boundary
        #loss = 100*mseloss(Cb,Cb_pred) #监督学习测试用
        loss.backward()                  
        optimizer.step()  
        train_mse += loss.item()   
    scheduler.step()
    train_mse /= len(train_loader)
    if i%100==0:
        print('epoch',i,': loss ',train_mse)

#训练集上计算解
C_pred = model(f_train).detach().cpu()
up_pred = np.zeros((ntrain,N,N))
for k in range(ntrain):
    interpolate_f_2d = interpolate.RegularGridInterpolator((np.linspace(0, 1, N),np.linspace(0, 1, N)), f[k])
    F = lambda x, y : interpolate_f_2d((x,y))
    C = C_pred[k]    
    for i in range(0,N):
        for j in range(0,N):
            x0 = (2*i+1)/(2*N)
            y0 = (2*j+1)/(2*N)
            f0 = F(x0,y0)
            c0 = c(x0,y0)
            mu0 = np.sqrt(c0)/eps
            c1 = C[4*N*j+4*i]
            c2 = C[4*N*j+4*i+1]
            c3 = C[4*N*j+4*i+2]
            c4 = C[4*N*j+4*i+3]
            up_pred[k,j,i] = f0/c0 + c1 + c2 + c3 + c4
rel_l2 = np.linalg.norm(up_pred - up_train) / np.linalg.norm(up_train)
print('relative l2 error on train data: ',rel_l2)
            
#测试集上计算解
C_pred = model(f_test).detach().cpu()
up_pred = np.zeros((ntest,N,N))
for k in range(ntest):
    interpolate_f_2d = interpolate.RegularGridInterpolator((np.linspace(0, 1, N),np.linspace(0, 1, N)), f[ntrain+k])
    F = lambda x, y : interpolate_f_2d((x,y))
    C = C_pred[k]    
    for i in range(0,N):
        for j in range(0,N):
            x0 = (2*i+1)/(2*N)
            y0 = (2*j+1)/(2*N)
            f0 = F(x0,y0)
            c0 = c(x0,y0)
            mu0 = np.sqrt(c0)/eps
            c1 = C[4*N*j+4*i]
            c2 = C[4*N*j+4*i+1]
            c3 = C[4*N*j+4*i+2]
            c4 = C[4*N*j+4*i+3]
            up_pred[k,j,i] = f0/c0 + c1 + c2 + c3 + c4


x = np.linspace(1/(2*N),1-1/(2*N),N)
xx,yy = np.meshgrid(x,x)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xx, yy, up_pred[k], cmap='rainbow')
ax.title.set_text('predicted solution u(x,y)')
plt.show()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xx, yy, up_test[k], cmap='rainbow')
ax.title.set_text('reference solution u(x,y)')
plt.show()
rel_l2 = np.linalg.norm(up_pred - up_test) / np.linalg.norm(up_test)
print('relative l2 error on test data: ',rel_l2)

#重构解
M = 10 #放大倍数
u_refine = np.zeros((M*N+1,M*N+1))
hh = 1/(M*N)
for i in range(0,N):
    for j in range(0,N):
        x0 = (2*i+1)/(2*N)
        y0 = (2*j+1)/(2*N)
        f0 = F(x0,y0)
        c0 = c(x0,y0)
        mu0 = np.sqrt(c0)/eps
        c1 = C[4*N*j+4*i]
        c2 = C[4*N*j+4*i+1]
        c3 = C[4*N*j+4*i+2]
        c4 = C[4*N*j+4*i+3]
        for ki in range(0,M+1):
            for kj in range(0,M+1):
                xhi = -1/(2*N) + ki*hh
                xhj = -1/(2*N) + kj*hh
                u_refine[j*M+kj,i*M+ki] = f0/c0 + c1*np.exp(mu0*xhi) + c2*np.exp(-mu0*xhi) + c3*np.exp(mu0*xhj) + c4*np.exp(-mu0*xhj) #仍然有大数*小数，造成误差
for l in range(0,M*N+1):
    s = l*hh
    u_refine[0,l] = b(s,0)
    u_refine[M*N,l] = b(s,1)
    u_refine[l,0] = b(0,s)
    u_refine[l,M*N] = b(1,s)
xh = np.linspace(0,1,N*M+1)
yh = np.linspace(0,1,N*M+1)
xxh,yyh = np.meshgrid(xh,yh)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xxh, yyh, u_refine, cmap='rainbow')
ax.title.set_text('refinement solution u(x,y)')
plt.show()






