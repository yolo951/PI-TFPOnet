
import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
from scipy import interpolate
from dim2_cnn import encoder_decoder
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
ntotal = ntrain + ntest
alpha = 1 #interface jump
beta = 0
eps = 1.0   # We multiply both sides of the equation by 1/eps, so eps here can be 1.0

epochs = 10000
learning_rate = 0.001
batch_size = 32
step_size = 2000
gamma = 0.5
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
f_train = torch.tensor(f_total[0:ntrain], dtype=torch.float32).to(device)
index_of_u_train = torch.LongTensor(index_of_u[0:ntrain]).to(device)
val_of_u_train = torch.tensor(val_of_u[0:ntrain], dtype=torch.float32).to(device)
B_train = torch.tensor(B_total[0:ntrain], dtype=torch.float32).to(device)
C_train = torch.tensor(C_total[0:ntrain], dtype=torch.float32).to(device)
up_train = up_total[0:ntrain]
# U_train = torch.tensor(U_total[0:ntrain], dtype=torch.float32).to(device)
f_test = torch.tensor(f_total[ntrain:ntotal], dtype=torch.float32).to(device)
up_test = torch.tensor(up_total[ntrain:ntotal], dtype=torch.float32).to(device)
C_test = torch.tensor(C_total[ntrain:ntotal], dtype=torch.float32).to(device)

C_pred = model(f_test).detach().cpu().reshape(f_test.shape[0], -1)
up_pred = np.zeros((ntest,N,N))

M = 8 # M-times test-resolution
up_refine = np.zeros((ntest, M*N+1,M*N+1))
ut_refine = np.zeros((ntest, M*N+1,M*N+1))
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
                    up_refine[k, j*M+kj,i*M+ki] = f0/c0 + c1p*np.exp(mu0*xhi) + c2p*np.exp(-mu0*xhi) + c3p*np.exp(mu0*xhj) + c4p*np.exp(-mu0*xhj)
                    ut_refine[k, j*M+kj,i*M+ki] = f0/c0 + c1t*np.exp(mu0*xhi) + c2t*np.exp(-mu0*xhi) + c3t*np.exp(mu0*xhj) + c4t*np.exp(-mu0*xhj)
    for l in range(0,M*N+1):
        s = l*hh
        up_refine[k, 0,l] = b(s,0)
        up_refine[k, M*N,l] = b(s,1)
        up_refine[k, l,0] = b(0,s)
        up_refine[k, l,M*N] = b(1,s)
        ut_refine[k, 0,l] = b(s,0)
        ut_refine[k, M*N,l] = b(s,1)
        ut_refine[k, l,0] = b(0,s)
        ut_refine[k, l,M*N] = b(1,s)

k = random.randrange(ntest) # select a random sample from the test dataset to show the error between the true value and the predicted value
# up_test = np.array(up_test.cpu())
# x = np.linspace(1/(2*N),1-1/(2*N),N)
# xx,yy = np.meshgrid(x,x)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(xx, yy, up_pred[k], cmap='rainbow')
# ax.title.set_text('predicted solution u(x,y)')
# # plt.show()
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(xx, yy, up_test[k], cmap='rainbow')
# ax.title.set_text('reference solution u(x,y)')
# # plt.show()
# rel_l2 = np.linalg.norm(up_pred - up_test) / np.linalg.norm(up_test)
# print('relative l2 error on test data: ',rel_l2)

# rel_l2 = np.linalg.norm(up_refine - ut_refine) / np.linalg.norm(ut_refine)
# rel_l_infty = np.linalg.norm((up_refine - ut_refine).flatten(), ord=np.inf) / np.linalg.norm(ut_refine.flatten(), ord=np.inf)
# print('relative l2 error on test data ({M}-times test-resolution): ',rel_l2)
# print('relative l_infty error on test data ({M}-times test-resolution): ',rel_l_infty)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# xh = np.linspace(0,1/2,int(N/2)*M)
# yh = np.linspace(0,1,N*M+1)
# xxh,yyh = np.meshgrid(xh,yh)
# ax.plot_surface(xxh, yyh, up_refine[k, :,0:int(N/2)*M], cmap='rainbow')
# xh = np.linspace(1/2,1,int(N/2)*M+1)
# yh = np.linspace(0,1,N*M+1)
# xxh,yyh = np.meshgrid(xh,yh)
# ax.plot_surface(xxh, yyh, up_refine[k, :,int(N/2)*M:N*M+1], cmap='rainbow')
# ax.title.set_text('refinement prediction u(x,y)')
# # plt.savefig(r'DeepONet-type\2d-smooth\2d_smooth_refine.png')
# #ax.view_init(elev=30, azim=-60)
# # plt.show()

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# xh = np.linspace(0,1/2,int(N/2)*M)
# yh = np.linspace(0,1,N*M+1)
# xxh,yyh = np.meshgrid(xh,yh)
# ax.plot_surface(xxh, yyh, ut_refine[k, :,0:int(N/2)*M], cmap='rainbow')
# xh = np.linspace(1/2,1,int(N/2)*M+1)
# yh = np.linspace(0,1,N*M+1)
# xxh,yyh = np.meshgrid(xh,yh)
# ax.plot_surface(xxh, yyh, ut_refine[k, :,int(N/2)*M:N*M+1], cmap='rainbow')
# ax.title.set_text('refinement ground truth u(x,y)')

# fig, ax = plt.subplots()
# xh = np.linspace(0,1,N*M+1)
# yh = np.linspace(0,1,N*M+1)
# xxh,yyh = np.meshgrid(xh,yh)
# cs = ax.contourf(xxh, yyh, np.abs(up_refine[k]-ut_refine[k]))
# cbar = fig.colorbar(cs)
# plt.title('error distribution')
# # plt.savefig(r'DeepONet-type\2d-smooth\2d_smooth_error.png')
# # plt.show()

# fig, ax = plt.subplots()
# xh = np.linspace(0,1,N*M+1)
# yh = np.linspace(0,1,N*M+1)
# xxh,yyh = np.meshgrid(xh,yh)
# cs = ax.contourf(xxh, yyh, ut_refine[k])
# cbar = fig.colorbar(cs)
# # plt.savefig(r'DeepONet-type\2d-smooth\2d_smooth_ground.png')
# plt.title('ground truth')
# plt.show()


xh = np.linspace(0,1,N*M+1)
yh = np.linspace(0,1,N*M+1)
xxh,yyh = np.meshgrid(xh,yh)

fig = plt.figure(figsize=(12, 3.5))
# [left, bottom, width, height]
ax0 = fig.add_axes([0.05, 0.1, 0.25, 0.8])
ax1 = fig.add_axes([0.34, 0.1, 0.25, 0.8])
ax_cb = fig.add_axes([0.60, 0.1, 0.01, 0.8])
ax2 = fig.add_axes([0.68, 0.1, 0.25, 0.8])
ax_cb2 = fig.add_axes([0.94, 0.1, 0.01, 0.8])

vmin = min(up_refine[k].min(), ut_refine[k].min())
vmax = max(up_refine[k].max(), ut_refine[k].max())
levels = np.linspace(vmin, vmax, 100)
cs0 = ax0.contourf(xxh, yyh, up_refine[k], levels=levels, cmap='RdYlBu_r')
cs1 = ax1.contourf(xxh, yyh, ut_refine[k], levels=levels, cmap='RdYlBu_r')
cbar = fig.colorbar(cs0, cax=ax_cb, format='%.3f')
error = np.abs(up_refine[k]-ut_refine[k])
levels_error = np.linspace(error.min(), error.max(), 100)
cs2 = ax2.contourf(xxh, yyh, error, levels=levels_error, cmap='RdYlBu_r')
cbar2 = fig.colorbar(cs2, cax=ax_cb2, format='%.3f')

ax0.set_title('Refinement prediction', fontsize=14)
ax1.set_title('Ground Truth', fontsize=14)
ax2.set_title('Point-wise error', fontsize=14)

for ax in [ax0, ax1, ax2]:
    ax.set_aspect('equal')
plt.savefig(r'DeepONet-type\2d-smooth\2d_smooth_compare.png')
plt.show()