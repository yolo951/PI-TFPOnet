

import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
from scipy import interpolate
from dim2_cnn import encoder_decoder
device = torch.device('cuda:4') 

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
        return torch.hstack([torch.tensor(0*x + 1.0), torch.cos(1 * w * x), torch.cos(2 * w * x), torch.cos(4 * w * x),  torch.sin(1 * w * x), torch.sin(2 * w * x), torch.sin(4 * w * x)])
    
    def forward(self, x):
        x = self.input_encoding(x)
        out = self.layers(x)
        return out

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
    

    C = np.linalg.solve(U,B)
    up = np.zeros((N,N))
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

def c(x,y):
    if x < 1/2:
        a = 16
    else:
        a = 1
    return a*1000.0

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
eps = 1.0
# f = generate(samples = ntotal, out_dim=N, length_scale=1)
# f = 0.9 + 0.1*f
# np.save('f.npy', f.astype(np.float32))

f = np.load('f.npy')
f *= 1000

epochs = 5000
learning_rate = 0.002
batch_size = 32
step_size = 2000
gamma = 0.5
model = DNN([7*N**2,512,512,128,512,4*N**2])
# model = encoder_decoder().to(device)


k = 0 
x = np.linspace(1/(2*N),1-1/(2*N),N)
xx,yy = np.meshgrid(x,x)

# U_total = np.zeros((ntotal,4*N**2,4*N**2), dtype=np.float32)
# B_total = np.zeros((ntotal,4*N**2), dtype=np.float32)
# C_total = np.zeros((ntotal,4*N**2), dtype=np.float32)
# up_total = np.zeros((ntotal,N,N), dtype=np.float32)
# f_total = np.zeros((ntotal,N**2), dtype=np.float32)
# for k in range(ntotal):              #todo并行加速for large N.
#     U, B, C, up = tfpm2d(N,f[k])
#     U_total[k] = U
#     B_total[k] = B
#     C_total[k] = C
#     up_total[k] = up
#     f_total[k] = f[k].reshape(-1)
# np.save('f_total.npy', f_total/1000.)
# np.save('U_total.npy', U_total)
# np.save('B_total.npy', B_total)
# np.save('C_total.npy', C_total)
# np.save('up_total.npy', up_total)
f_total = np.load('f_total.npy')
# f_total = np.load('f.npy')
U_total = np.load('U_total.npy')
B_total = np.load('B_total.npy')
C_total = np.load('C_total.npy')
up_total = np.load('up_total.npy')
f_train = f_total[0:ntrain]
U_train = U_total[0:ntrain]
B_train = B_total[0:ntrain]
C_train = C_total[0:ntrain]
f_train = torch.Tensor(f_train).to(dtype=torch.float32, device=device)
U_train = torch.Tensor(U_train).to(dtype=torch.float32, device=device)
B_train = torch.Tensor(B_train).to(dtype=torch.float32, device=device)
# C_train = torch.Tensor(C_train).to(dtype=torch.float32, device=device)
up_test = torch.Tensor(up_total[ntrain:ntotal]).to(dtype=torch.float32, device=device)
# up_train = torch.Tensor(up_total[:ntrain]).to(dtype=torch.float32, device=device)
f_test = torch.tensor(f_total[ntrain:ntotal]).to(device)
C_test = torch.Tensor(C_total[ntrain:]).to(dtype=torch.float32, device=device)


def train():

    # model = DNN([7*N**2,512,512,128,512,4*N**2]).to(device)
    model = encoder_decoder().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    mseloss = torch.nn.MSELoss()
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_train, U_train, B_train), batch_size=32, shuffle=True)

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
    loss_history = []
    rel_l2_history = []
    rel_l_infty_history = []
    # initial test error
    u_pred = model(f_test).reshape(ntest, -1, 4).sum(axis=-1)
    u = C_test.reshape(ntest, -1, 4).sum(axis=-1)
    rel_l2 = torch.linalg.norm(u_pred.flatten() - u.flatten()) / torch.linalg.norm(up_test.flatten())
    rel_l_infty = torch.linalg.norm(u_pred.flatten() - u.flatten(), ord=torch.inf) / torch.linalg.norm(up_test.flatten(), ord=torch.inf)
    rel_l2_history.append(rel_l2.item())
    rel_l_infty_history.append(rel_l_infty.item())
    for epoch in range(epochs):
        train_mse = 0
        model.train()
        for fb, Ub, Bb in dataloader:
            optimizer.zero_grad()
            Cb_pred = model(fb)
            # Cb_pred = model(fb).reshape(fb.shape[0], -1)
            U_jump = torch.index_select(Ub, 1, index_jump)
            B_jump = torch.index_select(Bb, 1, index_jump)
            U_continuous = torch.index_select(Ub, 1, index_continuous)
            B_continuous = torch.index_select(Bb, 1, index_continuous)
            U_boundary = torch.index_select(Ub, 1, index_boundary)
            B_boundary = torch.index_select(Bb, 1, index_boundary)
            loss_jump = mseloss(torch.einsum('bri, bi->br', U_jump, Cb_pred), B_jump)
            loss_continuous = mseloss(torch.einsum('bri, bi->br', U_continuous, Cb_pred), B_continuous)
            loss_boundary = mseloss(torch.einsum('bri, bi->br', U_boundary, Cb_pred), B_boundary)
            loss = 5.0*loss_jump + 5.0*loss_continuous +10.0*loss_boundary
            loss.backward()                  
            optimizer.step()  
            train_mse += loss.item()   
        scheduler.step()
        train_mse /= len(dataloader)
        loss_history.append(train_mse)

        
        
        if (epoch+1)%100==0:
            u_pred = model(f_test).reshape(ntest, -1, 4).sum(axis=-1)
            u = C_test.reshape(ntest, -1, 4).sum(axis=-1)
            rel_l2 = torch.linalg.norm(u_pred.flatten() - u.flatten()) / torch.linalg.norm(up_test.flatten())
            rel_l_infty = torch.linalg.norm(u_pred.flatten() - u.flatten(), ord=torch.inf) / torch.linalg.norm(up_test.flatten(), ord=torch.inf)
            rel_l2_history.append(rel_l2.item())
            rel_l_infty_history.append(rel_l_infty.item())
            # print(50.0*loss_jump.item(), 50.0*loss_continuous.item(), 100.0*loss_boundary.item())
            print('epoch',epoch,': loss ',train_mse, 'rel_l2 ',rel_l2.item(), 'rel_l_infty ',rel_l_infty.item())
    torch.save(model.state_dict(), 'model.pt')
    np.save('loss_history.npy', loss_history)
    np.save('rel_l2_history.npy', rel_l2_history)
 
def main():
    train()

    # model = DNN([7*N**2,512,512,128,512,4*N**2]).to(device)
    # model = encoder_decoder().to(device)
    # model.load_state_dict(torch.load('model.pt'))
    
    

    M = 8
    up_refine = np.zeros((ntest, M*N+1,M*N+1))
    up_test = np.zeros((ntest, M*N+1,M*N+1))
    C_pred = model(f_test).detach().cpu()
    # C_pred = model(f_test).detach().cpu().reshape(f_train.shape[0], -1)

    hh = 1/(M*N)
    xh = np.linspace(0,1,N*M+1)
    yh = np.linspace(0,1,N*M+1)
    xxh,yyh = np.meshgrid(xh,yh)
    
    for k in range(ntest):
        interpolate_f_2d = interpolate.RegularGridInterpolator((np.linspace(0, 1, N),np.linspace(0, 1, N)), f[ntrain+k])
        F = lambda x, y : interpolate_f_2d((x,y))
        c_pred = C_pred[k] 
        C = C_total[ntrain+k]
        for i in range(0,N):
            for j in range(0,N):
                x0 = (2*i+1)/(2*N)
                y0 = (2*j+1)/(2*N)
                f0 = F(x0,y0)
                c0 = c(x0,y0)
                mu0 = np.sqrt(c0)/eps
                c1nn = c_pred[4*N*j+4*i].item()
                c2nn = c_pred[4*N*j+4*i+1].item()
                c3nn = c_pred[4*N*j+4*i+2].item()
                c4nn = c_pred[4*N*j+4*i+3].item()
                c1 = C[4*N*j+4*i]
                c2 = C[4*N*j+4*i+1]
                c3 = C[4*N*j+4*i+2]
                c4 = C[4*N*j+4*i+3]
                for ki in range(0,M+1):
                    for kj in range(0,M+1):
                        xhi = -1/(2*N) + ki*hh
                        xhj = -1/(2*N) + kj*hh
                        up_refine[k, j*M+kj, i*M+ki] = f0/c0 + c1nn*np.exp(mu0*xhi) + c2nn*np.exp(-mu0*xhi) + c3nn*np.exp(mu0*xhj) + c4nn*np.exp(-mu0*xhj)
                        up_test[k, j*M+kj, i*M+ki] = f0/c0 + c1*np.exp(mu0*xhi) + c2*np.exp(-mu0*xhi) + c3*np.exp(mu0*xhj) + c4*np.exp(-mu0*xhj)
        for l in range(0,M*N+1):
            s = l*hh
            up_refine[k, 0,l] = b(s,0)
            up_refine[k, M*N,l] = b(s,1)
            up_refine[k, l,0] = b(0,s)
            up_refine[k, l,M*N] = b(1,s)
            up_test[k, 0,l] = b(s,0)
            up_test[k, M*N,l] = b(s,1)
            up_test[k, l,0] = b(0,s)
            up_test[k, l,M*N] = b(1,s)
    
    
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(xxh, yyh, up_refine[k], cmap='rainbow')
        ax.title.set_text('refinement solution u(x,y)')
        plt.savefig(f'fig/refine{k}')
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.plot_surface(xxh, yyh, up_test[k], cmap='rainbow')
        # ax.title.set_text('reference solution u(x,y)')
        # plt.savefig(f'fig/exact{k}')
        
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        xh = np.linspace(0,1/2,int(N/2)*M)
        yh = np.linspace(0,1,N*M+1)
        xxh,yyh = np.meshgrid(xh,yh)
        ax.plot_surface(xxh, yyh, up_refine[k, :,0:int(N/2)*M], cmap='rainbow')
        xh = np.linspace(1/2,1,int(N/2)*M+1)
        yh = np.linspace(0,1,N*M+1)
        xxh,yyh = np.meshgrid(xh,yh)
        ax.plot_surface(xxh, yyh, up_refine[k, :,int(N/2)*M:N*M+1], cmap='rainbow')
        ax.title.set_text('refinement prediction u(x,y)')
        #ax.view_init(elev=30, azim=-60)
        plt.savefig(f'fig/pred_refine{k}')

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # xh = np.linspace(0,1,N*M+1)
        # yh = np.linspace(0,1,N*M+1)
        # xxh,yyh = np.meshgrid(xh,yh)
        # ax.plot_surface(xxh, yyh, np.abs(up_refine-up_test), cmap='rainbow')
        # ax.title.set_text('error distribution')
        # #ax.view_init(elev=30, azim=-60)
        # plt.savefig(f'fig/error_distribution{k}')

        fig, ax = plt.subplots()
        xh = np.linspace(0,1,N*M+1)
        yh = np.linspace(0,1,N*M+1)
        xxh,yyh = np.meshgrid(xh,yh)
        cs = ax.contourf(xxh, yyh, np.abs(up_refine[k]-up_test[k]))
        cbar = fig.colorbar(cs)
        plt.title('error distribution')
        plt.savefig(f'fig/error_distribution{k}')

        # fig, ax = plt.subplots()
        # xh = np.linspace(0,1,N*M+1)
        # yh = np.linspace(0,1,N*M+1)
        # xxh,yyh = np.meshgrid(xh,yh)
        # cs = ax.contourf(xxh, yyh, up_test[k])
        # cbar = fig.colorbar(cs)
        # plt.title('ground truth')
        # plt.savefig(f'fig/ground_truth{k}')

plt.figure()
loss_history = np.load('loss_history.npy')
plt.plot(loss_history)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(1e-5, 1e+2)
plt.yscale("log")
plt.savefig('2d_singular_loss')
plt.show()

plt.figure()
rel_l2_history = np.load('rel_l2_history.npy')
epochs = 10000
plt.plot(np.arange(0, epochs+1, 100), rel_l2_history, '-*')
plt.xlabel('epochs')
plt.ylabel('relative l2 error')
plt.ylim(1e-3, 1e+0)
plt.yscale("log")
plt.savefig('2d_singular_l2')
plt.show()

if __name__ == "__main__":
    main()


