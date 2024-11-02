import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
from scipy import interpolate
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





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
        return torch.hstack([torch.ones_like(x).to(device), torch.cos(1 * w * x), torch.cos(2 * w * x), torch.cos(4 * w * x),  torch.sin(1 * w * x), torch.sin(2 * w * x), torch.sin(4 * w * x)])
    
    def forward(self, x):
        # x = self.input_encoding(x)
        out = self.layers(x)
        return out

def c(x,y):
    if x < 1/2:
        a = 16
    else:
        a = 1
    return a 

def b(x,y):
    if x >= 1/2:
        a = 2*(1-x)
    else:
        a = 0
    return a

N = 16
ntrain = 1000  
ntest = 200
eps = 1.0
ntotal = ntrain + ntest

model = DNN([N**2,512,128,128,512,4*N**2]).to(device)
model.load_state_dict(torch.load(r'DeepONet-type\2d-smooth\model_state.pt'))



f = np.load(r'DeepONet-type\2d-smooth\f.npy')
f_total = np.load(r'DeepONet-type\2d-smooth\matrixf.npy')
# U_total = np.load(r'DeepONet-type\2d-smooth\matrixU.npy')
index_of_u = np.load(r'DeepONet-type\2d-smooth\index_of_u.npy')
val_of_u = np.load(r'DeepONet-type\2d-smooth\val_of_u.npy')
B_total = np.load(r'DeepONet-type\2d-smooth\vectorB.npy')
C_total = np.load(r'DeepONet-type\2d-smooth\vectorC.npy')
up_total = np.load(r'DeepONet-type\2d-smooth\matrixup.npy')
up_test = torch.tensor(up_total[ntrain:ntotal], dtype=torch.float32).to(device)
f_train = torch.tensor(f_total[0:ntrain], dtype=torch.float32).to(device)
f_test = torch.tensor(f_total[ntrain:ntotal], dtype=torch.float32).to(device)

C_pred = model(f_test).detach().cpu().reshape(f_test.shape[0], -1)
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
up_test = np.array(up_test.cpu())
k = 0
x = np.linspace(1/(2*N),1-1/(2*N),N)
xx,yy = np.meshgrid(x,x)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xx, yy, up_pred[k], cmap='rainbow')
ax.title.set_text('predicted solution u(x,y)')
# plt.show()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xx, yy, up_test[k], cmap='rainbow')
ax.title.set_text('reference solution u(x,y)')
M = 1
up_refine = np.zeros((M*N+1,M*N+1))
ut_refine = np.zeros((M*N+1,M*N+1))
Cp = C_pred[k]
Ct = C_total[ntrain+k]
hh = 1/(M*N)
for i in range(0,N):
    for j in range(0,N):
        x0 = (2*i+1)/(2*N)
        y0 = (2*j+1)/(2*N)
        f0 = F(x0,y0)
        c0 = c(x0,y0)
        mu0 = np.sqrt(c0)/eps
        c1p = Cp[4*N*j+4*i]
        c2p = Cp[4*N*j+4*i+1]
        c3p = Cp[4*N*j+4*i+2]
        c4p = Cp[4*N*j+4*i+3]
        c1t = Ct[4*N*j+4*i]
        c2t = Ct[4*N*j+4*i+1]
        c3t = Ct[4*N*j+4*i+2]
        c4t = Ct[4*N*j+4*i+3]
        for ki in range(0,M):
            for kj in range(0,M):
                xhi = -1/(2*N) + ki*hh
                xhj = -1/(2*N) + kj*hh
                up_refine[j*M+kj,i*M+ki] = f0/c0 + c1p*np.exp(mu0*xhi) + c2p*np.exp(-mu0*xhi) + c3p*np.exp(mu0*xhj) + c4p*np.exp(-mu0*xhj) #仍然有大数*小数，造成误差
                ut_refine[j*M+kj,i*M+ki] = f0/c0 + c1t*np.exp(mu0*xhi) + c2t*np.exp(-mu0*xhi) + c3t*np.exp(mu0*xhj) + c4t*np.exp(-mu0*xhj) #仍然有大数*小数，造成误差
for l in range(0,M*N+1):
    s = l*hh
    up_refine[0,l] = b(s,0)
    up_refine[M*N,l] = b(s,1)
    up_refine[l,0] = b(0,s)
    up_refine[l,M*N] = b(1,s)
    ut_refine[0,l] = b(s,0)
    ut_refine[M*N,l] = b(s,1)
    ut_refine[l,0] = b(0,s)
    ut_refine[l,M*N] = b(1,s)
fig, ax = plt.subplots()
xh = np.linspace(0,1,N*M+1)
yh = np.linspace(0,1,N*M+1)
xxh,yyh = np.meshgrid(xh,yh)
cs = ax.contourf(xxh, yyh, np.abs(up_refine-ut_refine))
cbar = fig.colorbar(cs)
plt.title('error distribution')
# plt.savefig(r'DeepONet-type\2d-smooth\2d_smooth_error.png')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
xh = np.linspace(0,1/2,int(N/2)*M)
yh = np.linspace(0,1,N*M+1)
xxh,yyh = np.meshgrid(xh,yh)
ax.plot_surface(xxh, yyh, ut_refine[:,0:int(N/2)*M], cmap='rainbow')
xh = np.linspace(1/2,1,int(N/2)*M+1)
yh = np.linspace(0,1,N*M+1)
xxh,yyh = np.meshgrid(xh,yh)
ax.plot_surface(xxh, yyh, ut_refine[:,int(N/2)*M:N*M+1], cmap='rainbow')
ax.title.set_text('ground truth u(x,y)')
# plt.savefig(r'DeepONet-type\2d-smooth\2d_smooth_ground.png')
plt.title('ground truth')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
xh = np.linspace(0,1/2,int(N/2)*M)
yh = np.linspace(0,1,N*M+1)
xxh,yyh = np.meshgrid(xh,yh)
ax.plot_surface(xxh, yyh, up_refine[:,0:int(N/2)*M], cmap='rainbow')
xh = np.linspace(1/2,1,int(N/2)*M+1)
yh = np.linspace(0,1,N*M+1)
xxh,yyh = np.meshgrid(xh,yh)
ax.plot_surface(xxh, yyh, up_refine[:,int(N/2)*M:N*M+1], cmap='rainbow')
ax.title.set_text('refinement prediction u(x,y)')
# plt.savefig(r'DeepONet-type\2d-smooth\2d_smooth_refine.png')
plt.show()

