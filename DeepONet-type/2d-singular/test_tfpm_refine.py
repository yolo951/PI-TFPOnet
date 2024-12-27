

import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
from scipy import sparse
from scipy.sparse import linalg
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

def c(x,y):
    if x < 1/2:
        a = 16
    else:
        a = 1
    return a*1000.0

def b(x,y):
    if x >= 1/2:
        a = 1-x
    else:
        a = -x
    return a
    
def tfpm2d(N, f, eps=1.0, alpha=1.0, beta=0.0): 
    h = 1/(2*N)
    index = np.zeros((4*N**2, 8))
    val = np.zeros((4*N**2, 8))
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
        index[i, :4] = np.array([4*i,4*i+1,4*i+2,4*i+3])
        val[i, :4] = np.array([np.exp(-mu0*h),np.exp(-mu0*h),np.exp(-2*mu0*h),1])
        B[i] = (- f0/c0 + b(xi,0))*np.exp(-mu0*h)
    #boundary y=1.
    for i in range(0,N):
        x0 = (2*i+1)*h
        y0 = 1-h
        f0 = F(x0,y0)
        c0 = c(x0,y0)
        mu0 = np.sqrt(c0)/eps
        xi = (2*i+1)*h
        index[N+i, :4] = np.array([4*N*(N-1)+4*i,4*N*(N-1)+4*i+1,4*N*(N-1)+4*i+2,4*N*(N-1)+4*i+3])
        val[N+i, :4] = np.array([np.exp(-mu0*h),np.exp(-mu0*h),1,np.exp(-2*mu0*h)])
        B[N+i] = (- f0/c0 + b(xi,1))*np.exp(-mu0*h)
    #boundary x=0.
    for i in range(0,N):
        x0 = h
        y0 = (2*i+1)*h
        f0 = F(x0,y0)
        c0 = c(x0,y0)
        mu0 = np.sqrt(c0)/eps
        yi = (2*i+1)*h
        index[2*N+i, :4] = np.array([4*N*i,4*N*i+1,4*N*i+2,4*N*i+3])
        val[2*N+i, :4] = np.array([np.exp(-2*mu0*h),1,np.exp(-mu0*h),np.exp(-mu0*h)])
        B[2*N+i] = (- f0/c0 + b(0,yi))*np.exp(-mu0*h)
    #boundary x=1.
    for i in range(0,N):
        x0 = 1-h
        y0 = (2*i+1)*h
        f0 = F(x0,y0)
        c0 = c(x0,y0)
        mu0 = np.sqrt(c0)/eps
        yi = (2*i+1)*h
        index[3*N+i, :4] = np.array([4*N*(i+1)-4,4*N*(i+1)-3,4*N*(i+1)-2,4*N*(i+1)-1])
        val[3*N+i, :4] = np.array([1,np.exp(-2*mu0*h),np.exp(-mu0*h),np.exp(-mu0*h)])
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
            index[4*N+N*i+j] = np.array([4*N*j+4*i,4*N*j+4*i+1,4*N*j+4*i+2,4*N*j+4*i+3,4*N*j+4*i+4,4*N*j+4*i+5,4*N*j+4*i+6,4*N*j+4*i+7])
            val[4*N+N*i+j] = np.array([np.exp((mu0-mu)*h),np.exp(-(mu0+mu)*h),np.exp(-mu*h),np.exp(-mu*h),-np.exp(-(mu1+mu)*h),-np.exp((mu1-mu)*h),-np.exp(-mu*h),-np.exp(-mu*h)])
            B[4*N+N*i+j] = (f1/c1 - f0/c0)*np.exp(-mu*h)
            index[4*N+N*(N-1)+N*i+j] = np.array([4*N*j+4*i,4*N*j+4*i+1,4*N*j+4*i+2,4*N*j+4*i+3,4*N*j+4*i+4,4*N*j+4*i+5,4*N*j+4*i+6,4*N*j+4*i+7])
            val[4*N+N*(N-1)+N*i+j] = np.array([mu0*np.exp((mu0-mu)*h),-mu0*np.exp(-(mu0+mu)*h),0,0,-mu1*np.exp(-(mu1+mu)*h),mu1*np.exp((mu1-mu)*h),0,0])
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
            index[4*N+2*N*(N-1)+N*j+i] = np.array([4*N*j+4*i,4*N*j+4*i+1,4*N*j+4*i+2,4*N*j+4*i+3,4*N*(j+1)+4*i,4*N*(j+1)+4*i+1,4*N*(j+1)+4*i+2,4*N*(j+1)+4*i+3])
            val[4*N+2*N*(N-1)+N*j+i] = np.array([np.exp(-mu*h),np.exp(-mu*h),np.exp((mu0-mu)*h),np.exp(-(mu0+mu)*h),-np.exp(-mu*h),-np.exp(-mu*h),-np.exp(-(mu1+mu)*h),-np.exp((mu1-mu)*h)])
            B[4*N+2*N*(N-1)+N*j+i] = (f1/c1 - f0/c0)*np.exp(-mu*h)
            index[4*N+3*N*(N-1)+N*j+i] = np.array([4*N*j+4*i,4*N*j+4*i+1,4*N*j+4*i+2,4*N*j+4*i+3,4*N*(j+1)+4*i,4*N*(j+1)+4*i+1,4*N*(j+1)+4*i+2,4*N*(j+1)+4*i+3])
            val[4*N+3*N*(N-1)+N*j+i] = np.array([0,0,mu0*np.exp((mu0-mu)*h),-mu0*np.exp(-(mu0+mu)*h),0,0,-mu1*np.exp(-(mu1+mu)*h),mu1*np.exp((mu1-mu)*h)])
            B[4*N+3*N*(N-1)+N*j+i] = 0
    
    #计算解
    U_data = val.flatten()
    U_row = np.repeat(np.arange(4*N**2), 8)
    U_col = index.flatten()
    U_sparse = sparse.csr_matrix((U_data, (U_row, U_col)), shape=(4*N**2, 4*N**2))
    C = linalg.spsolve(U_sparse, B)
    # up = np.zeros((N,N)) #每个网格中心点值，不包含边界
    up = np.zeros((N+1, N+1))
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
            xhi = -1/(2*N)
            xhj = -1/(2*N)
            up[j,i] = f0/c0 + c1*np.exp(mu0*xhi) + c2*np.exp(-mu0*xhi) + c3*np.exp(mu0*xhj) + c4*np.exp(-mu0*xhj)
    for l in range(0, N+1):
        s = l/N
        up[0,l] = b(s,0)
        up[N,l] = b(s,1)
        up[l,0] = b(0,s)
        up[l,N] = b(1,s)
    return B, C, up, index, val

if __name__ == '__main__':
    N = 32
    ntotal = 1
    alpha = 1 # interface jump
    beta = 0
    eps = 1.0  # We multiply both sides of the equation by 1/eps, so eps here can be 1.0
    f = generate(samples = ntotal, out_dim=N+1, length_scale=1)
    f *= 1000.0

    k = 0 
    x = np.linspace(1/(2*N),1-1/(2*N),N)
    xx,yy = np.meshgrid(x,x)

    # tfpm on sparse grid
    B, C, up, idx, v = tfpm2d(N,f[k])
    # compute interpolation of up(0.5, y)
    interp_func = interpolate.interp1d(np.linspace(0, 1, N+1), up[:, int(N/2)])
    # refine on 8-times fine grid
    M = 4
    interpolate_f_2d = interpolate.RegularGridInterpolator((np.linspace(0, 1, N+1),np.linspace(0, 1, N+1)), f[0])
    F = lambda x, y : interpolate_f_2d((x, y))
    hh = 1/(M*N)
    up_refine = np.zeros((M*N+1,M*N+1))
    for i in range(0,N):
        for j in range(0,N):
            x0 = (2*i+1)/(2*N)
            y0 = (2*j+1)/(2*N)
            f0 = F(x0,y0)
            c0 = c(x0,y0)
            mu0 = np.sqrt(c0)/eps
            c1p = C[4*N*j+4*i]
            c2p = C[4*N*j+4*i+1]
            c3p = C[4*N*j+4*i+2]
            c4p = C[4*N*j+4*i+3]
            for ki in range(0,M):
                for kj in range(0,M):
                    xhi = -1/(2*N) + ki*hh
                    xhj = -1/(2*N) + kj*hh
                    up_refine[j*M+kj,i*M+ki] = f0/c0 + c1p*np.exp(mu0*xhi) + c2p*np.exp(-mu0*xhi) + c3p*np.exp(mu0*xhj) + c4p*np.exp(-mu0*xhj)
    for l in range(0,M*N+1):
        s = l*hh
        up_refine[0,l] = b(s,0)
        up_refine[M*N,l] = b(s,1)
        up_refine[l,0] = b(0,s)
        up_refine[l,M*N] = b(1,s)

    # directly calculate on fine grid
    grid_fine = np.linspace(0,1,N*M+1)
    X, Y = np.meshgrid(grid_fine, grid_fine)
    points = np.stack((Y.flatten(), X.flatten()), axis=-1)
    f_fine = interpolate_f_2d(points).reshape(N*M+1, N*M+1)
    B, C, up_fine, idx, v = tfpm2d(N*M, f_fine)

    # make plots
    # generate plot at x=0.5
    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # for i in range(3):
    #     y_sparse = np.linspace(0, 1, N+1)
    #     y_fine = np.linspace(0, 1, N*M+1)
    #     uu_refine = up_refine[:, int(N*M/2)+M*i]
    #     uu_fine = up_fine[:, int(N*M/2)+M*i]
    #     # uu_interp = interp_func(yy)
    #     axes[i].plot(y_fine, uu_refine, '-*', label='refine')
    #     axes[i].plot(y_fine, uu_fine, '.-', label='fine')
    #     axes[i].legend()
    #     axes[i].set_title('x={}'.format(0.5+i*1/N/M))
        # plt.plot(y_sparse, uu_refine, '-*', label='refine')
        # plt.plot(y_fine, uu_fine, '.-', label='fine')
        # # plt.plot(y_fine, uu_interp, '-*', label='interp')
        # plt.legend()
        # plt.title('x={}'.format(0.5+i*1/N))


    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # xh = np.linspace(0,1/2,int(N/2)*M+1)
    # yh = np.linspace(0,1,N*M+1)
    # xxh,yyh = np.meshgrid(xh[:-1],yh)
    # ax.plot_surface(xxh, yyh, up_refine[:,0:int(N/2)*M], cmap='rainbow')
    # xh = np.linspace(1/2,1,int(N/2)*M+1)
    # yh = np.linspace(0,1,N*M+1)
    # xxh,yyh = np.meshgrid(xh,yh)
    # ax.plot_surface(xxh, yyh, up_refine[:,int(N/2)*M:], cmap='rainbow')
    # ax.title.set_text('refinement u(x,y)')

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # xh = np.linspace(0,1/2,int(N/2)*M+1)
    # yh = np.linspace(0,1,N*M+1)
    # xxh,yyh = np.meshgrid(xh[:-1],yh)
    # ax.plot_surface(xxh, yyh, up_fine[:,0:int(N/2)*M], cmap='rainbow')
    # xh = np.linspace(1/2,1,int(N/2)*M+1)
    # yh = np.linspace(0,1,N*M+1)
    # xxh,yyh = np.meshgrid(xh,yh)
    # ax.plot_surface(xxh, yyh, up_fine[:,int(N/2)*M:], cmap='rainbow')
    # ax.title.set_text('fine grid ground truth u(x,y)')

    xh = np.linspace(0,1,N*M+1)
    yh = np.linspace(0,1,N*M+1)
    xxh1, yyh1 = np.meshgrid(xh[:N*M//2], yh)
    xxh2, yyh2 = np.meshgrid(xh[N*M//2+1:], yh)

    fig = plt.figure(figsize=(12, 3.5))
    # [left, bottom, width, height]
    ax0 = fig.add_axes([0.05, 0.1, 0.25, 0.8])
    ax1 = fig.add_axes([0.34, 0.1, 0.25, 0.8])
    ax_cb = fig.add_axes([0.60, 0.1, 0.01, 0.8])
    ax2 = fig.add_axes([0.68, 0.1, 0.25, 0.8])
    ax_cb2 = fig.add_axes([0.94, 0.1, 0.01, 0.8])

    vmin = min(up_refine.min(), up_fine.min())
    vmax = max(up_refine.max(), up_fine.max())
    levels = np.linspace(vmin, vmax, 100)
    cs0 = ax0.contourf(xxh1, yyh1, up_refine[:, :N*M//2], levels=levels, cmap='RdYlBu_r')
    ax0.contourf(xxh2, yyh2, up_refine[:, N*M//2+1:], levels=levels, cmap='RdYlBu_r')
    cs1 = ax1.contourf(xxh1, yyh1, up_fine[:, :N*M//2], levels=levels, cmap='RdYlBu_r')
    ax1.contourf(xxh2, yyh2, up_fine[:, N*M//2+1:], levels=levels, cmap='RdYlBu_r')
    cbar = fig.colorbar(cs0, cax=ax_cb, format='%.3f')
    error = np.abs(up_fine-up_refine)
    error = np.hstack((error[:, :N*M//2], error[:, N*M//2+1:]))
    levels_error = np.linspace(error.min(), error.max(), 100)
    cs2 = ax2.contourf(xxh1, yyh1, error[:, :N*M//2], levels=levels_error, cmap='RdYlBu_r')
    ax2.contourf(xxh2, yyh2, error[:, N*M//2:], levels=levels_error, cmap='RdYlBu_r')
    cbar2 = fig.colorbar(cs2, cax=ax_cb2, format='%.3f')

    ax0.set_title('Refinement prediction', fontsize=14)
    ax1.set_title('Ground Truth', fontsize=14)
    ax2.set_title('Point-wise error', fontsize=14)

    for ax in [ax0, ax1, ax2]:
        ax.set_aspect('equal')
    # plt.savefig(r'DeepONet-type\2d-singular\2d_singular_compare.png')
    plt.show()




