

import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections import OrderedDict
from scipy import interpolate
from timeit import default_timer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(samples=10, begin=0, end=1, random_dim=101, out_dim=101, length_scale=1, interp="cubic", A=0):
    space = GRF(begin, end, length_scale=length_scale, N=random_dim, interp=interp)
    features = space.random(samples, A)
    features = features.reshape((features.shape[0], random_dim, random_dim))
    x_grid = np.linspace(begin, end, out_dim)
    x_data = space.eval_u(features, x_grid, x_grid)
    return x_data

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
    
class DeepONet(nn.Module):
    def __init__(self,b_dim,t_dim):
        super(DeepONet, self).__init__()
        self.b_dim = b_dim
        self.t_dim = t_dim
        
        self.branch = nn.Sequential(
            nn.Linear(self.b_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
        )
        
        self.trunk = nn.Sequential(
            nn.Linear(self.t_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
        )
        
        self.b = Parameter(torch.zeros(1))

        
    def forward(self, x, l):
        x = self.branch(x)
        l = self.trunk(l)
        
        res = torch.einsum("bi,bki->bk", x, l)
        res = res.unsqueeze(-1) + self.b
        return res
    
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
    return up



N = 16
half_N = int(N/2)
ntrain = 1000 
ntest = 200
ntotal = ntrain + ntest
alpha = 1 #interface jump
beta = 0
eps = 1.001
# f_total = generate(samples = ntotal, out_dim=N, length_scale=1)
# f_total = 1. + 0.1*f_total
# np.save('f.npy', f_total)
# f_total = np.load('f_total.npy')


type_ = 'unsupervised' # supervised
epochs = 2000
learning_rate = 0.001
batch_size = 10
step_size = 1000
gamma = 0.5
model0 = DeepONet(half_N*N,2).to(device)
model1 = DeepONet(half_N*N,2).to(device)
optimizer0 = torch.optim.Adam(model0.parameters(), lr=learning_rate, weight_decay=1e-4)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler0 = torch.optim.lr_scheduler.StepLR(optimizer0, step_size=step_size, gamma=gamma)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=step_size, gamma=gamma)


# up_total = np.zeros((ntotal, N, N))
# loc_total = np.zeros((ntotal,N**2, 2))
# for k in range(ntotal):              #todo并行加速for large N.
#     up = tfpm2d(N,f_total[k])
#     up_total[k] = up
# np.save('f_total.npy', f_total)
# np.save('up_total', up_total)
f_total = np.load('f_total.npy')
up_total = np.load('up_total.npy')



f0 = f_total[:, :, :half_N]
f1 = f_total[:, :, half_N:]
x = np.linspace(1/(2*N),1-1/(2*N),N)
x0, x1 = x[:half_N], x[half_N:]
xx, yy = np.meshgrid(x0, x)
input_loc0 =  np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
input_loc0 = np.tile(np.expand_dims(input_loc0,axis=0),(ntotal,1,1))
xx, yy = np.meshgrid(x1, x)
input_loc1 = np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
input_loc1 = np.tile(np.expand_dims(input_loc1,axis=0),(ntotal,1,1))
u0 = up_total[:, :, :half_N]
u1 = up_total[:, :, half_N:]

f_train0 = torch.tensor(f0[:ntrain].reshape(ntrain,-1), dtype=torch.float32).to(device)
f_train1 = torch.tensor(f1[:ntrain].reshape(ntrain,-1), dtype=torch.float32).to(device)
input_loc0 = torch.tensor(input_loc0[:ntrain], dtype=torch.float32).to(device)
input_loc1 = torch.tensor(input_loc1[:ntrain], dtype=torch.float32).to(device)
u_train0 = torch.tensor(u0[:ntrain].reshape(ntrain,-1), dtype=torch.float32).to(device)
u_train1 = torch.tensor(u1[:ntrain].reshape(ntrain,-1), dtype=torch.float32).to(device)
 
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_train0,input_loc0,u_train0, f_train1,input_loc1,u_train1), batch_size=batch_size, shuffle=True)
mseloss = torch.nn.MSELoss(reduction='mean')
mse_history = []

x_b = torch.zeros((1,4*N,2), dtype=torch.float32).to(device)
y_b = torch.zeros((1,4*N,1), dtype=torch.float32).to(device)
for k in range(N):
    x_b[0,k] = torch.tensor([1/(2*N)+k/N,1/(2*N)])
    x_b[0,N+k] = torch.tensor([1-1/(2*N),1/(2*N)+k/N])
    x_b[0,2*N+k] = torch.tensor([1/(2*N)+k/N,1-1/(2*N)])
    x_b[0,3*N+k] = torch.tensor([1/(2*N),1/(2*N)+k/N])
    y_b[0,k] = torch.tensor([b(1/(2*N)+k/N,0)])
    y_b[0,N+k] = torch.tensor([b(1,1/(2*N)+k/N)])
    y_b[0,2*N+k] = torch.tensor([b(1/(2*N)+k/N,1)])
    y_b[0,3*N+k] = torch.tensor([b(0,1/(2*N)+k/N)])
x_b = x_b.repeat([batch_size,1,1])
y_b = y_b.repeat([batch_size,1,1])
xr_i = torch.zeros((1,N,2), dtype=torch.float32).to(device)
xl_i = torch.zeros((1,N,2), dtype=torch.float32).to(device)
for k in range(N):
    xr_i[0,k] = torch.tensor([1/2+0.01,1/(2*N)+k/N])
    xl_i[0,k] = torch.tensor([1/2-0.01,1/(2*N)+k/N])
xr_i = xr_i.repeat([batch_size,1,1])
xl_i = xl_i.repeat([batch_size,1,1])

for ep in range(epochs):
    model0.train()
    model1.train()
    t1 = default_timer()
    train_mse = 0
    for ff0, l0, y0, ff1, l1, y1 in train_loader:
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        x00 = l0[:,:,0:1]
        x01 = l0[:,:,1:2]
        x10 = l1[:,:,0:1]
        x11 = l1[:,:,1:2]
        x00.requires_grad_(True)
        x01.requires_grad_(True)
        x10.requires_grad_(True)
        x11.requires_grad_(True)
        y_pred0 = model0(ff0,torch.cat((x00, x01), dim=2))
        y_pred1 = model1(ff1,torch.cat((x10, x11), dim=2))
        if type_ == 'supervised':
            mse  = 100.0*mseloss(y_pred0.flatten(), y0.flatten())+100.0*mseloss(y_pred1.flatten(), y1.flatten())
        else:
            y_x1 = torch.autograd.grad(y_pred0, x00, grad_outputs=torch.ones_like(y_pred0), create_graph=True)[0]
            y_x1x1 = torch.autograd.grad(y_x1, x00, grad_outputs=torch.ones_like(y_x1), create_graph=True)[0]
            y_x2 = torch.autograd.grad(y_pred0, x01, grad_outputs=torch.ones_like(y_pred0), create_graph=True)[0]
            y_x2x2 = torch.autograd.grad(y_x2, x01, grad_outputs=torch.ones_like(y_x2), create_graph=True)[0]
            # Cx1x2 = torch.gt(x00,0.5)*16+(~torch.gt(x00,0.5))*1
            Cx1x2 = torch.where(x00<=0.5, 16.0, 1.0)
            F = - 0.001*(y_x1x1 + y_x2x2) + Cx1x2*y_pred0 - ff0.unsqueeze(2)
            mse_f0 = torch.mean(F ** 2)
            
            y_x1 = torch.autograd.grad(y_pred1, x10, grad_outputs=torch.ones_like(y_pred1), create_graph=True)[0]
            y_x1x1 = torch.autograd.grad(y_x1, x10, grad_outputs=torch.ones_like(y_x1), create_graph=True)[0]
            y_x2 = torch.autograd.grad(y_pred1, x11, grad_outputs=torch.ones_like(y_pred1), create_graph=True)[0]
            y_x2x2 = torch.autograd.grad(y_x2, x11, grad_outputs=torch.ones_like(y_x2), create_graph=True)[0]
            # Cx1x2 = torch.gt(x10,0.5)*16+(~torch.gt(x10,0.5))*1
            Cx1x2 = torch.where(x10<=0.5, 16.0, 1.0)
            F = - 0.001*(y_x1x1 + y_x2x2) + Cx1x2*y_pred1 - ff1.unsqueeze(2)
            mse_f1 = torch.mean(F ** 2)

            y_bp0 = model0(ff0,x_b)
            mse_b0 = mseloss(y_b,y_bp0)
            y_bp1 = model1(ff1,x_b)
            mse_b1 = mseloss(y_b,y_bp1)
            
            # yr = model1(ff1,xr_i)
            # yl = model0(ff0,xl_i).to(device1)
            # mse_i = mseloss(yr,yl+1)
            mse0 = 100*mse_f0+100*mse_b0
            mse1 = 100*mse_f1 + 100*mse_b1 #+mse_i
        mse.backward()
        optimizer0.step()
        optimizer1.step()
        train_mse += mse.item()
    scheduler0.step()
    scheduler1.step()
    train_mse /= len(train_loader)
    t2 = default_timer()
    mse_history.append(train_mse)
    if ep%100==0:
        # print((mse_f0 + mse_f0),100*(mse_b0+mse_b1),mse_i)
        print('\repoch {:d}/{:d} , MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse, t2 - t1), end='', flush=True)


f_test0 = torch.tensor(f0[-ntest:].reshape(ntest,-1), dtype=torch.float32).to(device)
f_test1 = torch.tensor(f1[-ntest:].reshape(ntest,-1), dtype=torch.float32).to(device)
input_loc0 = torch.tensor(input_loc0[-ntest:]).to(device)
input_loc1 = torch.tensor(input_loc1[-ntest:]).to(device)
u_test = up_total[-ntest:]



index = 0
test_mse = 0
with torch.no_grad():
    out0 = model0(f_test0, input_loc0).reshape(ntest, N, half_N)
    out1 = model1(f_test1, input_loc1).reshape(ntest, N, half_N)
    pred = torch.concat((out0, out1), dim=-1)
    pred = pred.detach().cpu().numpy()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    xh = np.linspace(1/(2*N),1-1/(2*N),N)
    yh = np.linspace(1/(2*N),1-1/(2*N),N)
    xxh,yyh = np.meshgrid(xh,yh)
    ax.plot_surface(xxh, yyh, np.abs(pred[0].reshape(N,N)), cmap='rainbow')
    ax.title.set_text('test')
    #ax.view_init(elev=30, azim=-60)
    plt.savefig('test')
    print('test error on high resolution: relative L2 norm = ', np.linalg.norm(pred.flatten()-u_test.flatten()) / np.linalg.norm(u_test.flatten()))
    print('test error on high resolution: relative L_inf norm = ', np.linalg.norm(pred.flatten()-u_test.flatten(), ord=np.inf) / np.linalg.norm(u_test.flatten(), ord=np.inf))


