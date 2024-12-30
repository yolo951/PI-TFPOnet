
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
    if (0.25<=y<=0.5 and 0.25<=x<=0.75) or (0.5<=y<=0.75 and 0.25<=x<=0.5):
        a = 1
    else:
        a = 16
    return a*1000.0

def b(x,y):
    if y==1:
        a = 1-x
    elif y==0:
        a = x
    elif x==1:
        a = 1-y
    else:
        a = y
    return a*0.5
    
def tfpm2d(N, f, interface, alpha=1.0, beta=0.0, eps=1.0): 
    h = 1/(2*N)
    index = np.zeros((4*N**2, 8))
    val = np.zeros((4*N**2, 8))
    B = np.zeros(4*N**2)
    interpolate_f_2d = interpolate.RegularGridInterpolator((np.linspace(0, 1, f.shape[-1]),np.linspace(0, 1, f.shape[-1])), f)
    F = lambda x, y : interpolate_f_2d((x,y))
    left, right, buttom, top = interface['vertical_left'], interface['vertical_right'], interface['horizontal_buttom'], interface['horizontal_top']
    index_jump = []
    
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
            # continuous condition of u
            index[4*N+N*i+j] = np.array([4*N*j+4*i,4*N*j+4*i+1,4*N*j+4*i+2,4*N*j+4*i+3,4*N*j+4*i+4,4*N*j+4*i+5,4*N*j+4*i+6,4*N*j+4*i+7])
            val[4*N+N*i+j] = np.array([np.exp((mu0-mu)*h),np.exp(-(mu0+mu)*h),np.exp(-mu*h),np.exp(-mu*h),-np.exp(-(mu1+mu)*h),-np.exp((mu1-mu)*h),-np.exp(-mu*h),-np.exp(-mu*h)])
            B[4*N+N*i+j] = (f1/c1 - f0/c0)*np.exp(-mu*h)
            # continuous condition of du
            index[4*N+N*(N-1)+N*i+j] = np.array([4*N*j+4*i,4*N*j+4*i+1,4*N*j+4*i+2,4*N*j+4*i+3,4*N*j+4*i+4,4*N*j+4*i+5,4*N*j+4*i+6,4*N*j+4*i+7])
            val[4*N+N*(N-1)+N*i+j] = np.array([mu0*np.exp((mu0-mu)*h),-mu0*np.exp(-(mu0+mu)*h),0,0,-mu1*np.exp(-(mu1+mu)*h),mu1*np.exp((mu1-mu)*h),0,0])
            B[4*N+N*(N-1)+N*i+j] = 0
            #interface
            if (i, j) in left:
                B[4*N+N*i+j] = B[4*N+N*i+j] - alpha*np.exp(-mu*h)
                B[4*N+N*(N-1)+N*i+j] = B[4*N+N*(N-1)+N*i+j] - beta*np.exp(-mu*h)
                index_jump.extend([4*N+N*i+j, 4*N+N*(N-1)+N*i+j])
            elif (i, j) in right:
                B[4*N+N*i+j] = B[4*N+N*i+j] + alpha*np.exp(-mu*h)
                B[4*N+N*(N-1)+N*i+j] = B[4*N+N*(N-1)+N*i+j] + beta*np.exp(-mu*h)
                index_jump.extend([4*N+N*i+j, 4*N+N*(N-1)+N*i+j])

    
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
            # continuous condition of u
            index[4*N+2*N*(N-1)+N*j+i] = np.array([4*N*j+4*i,4*N*j+4*i+1,4*N*j+4*i+2,4*N*j+4*i+3,4*N*(j+1)+4*i,4*N*(j+1)+4*i+1,4*N*(j+1)+4*i+2,4*N*(j+1)+4*i+3])
            val[4*N+2*N*(N-1)+N*j+i] = np.array([np.exp(-mu*h),np.exp(-mu*h),np.exp((mu0-mu)*h),np.exp(-(mu0+mu)*h),-np.exp(-mu*h),-np.exp(-mu*h),-np.exp(-(mu1+mu)*h),-np.exp((mu1-mu)*h)])
            B[4*N+2*N*(N-1)+N*j+i] = (f1/c1 - f0/c0)*np.exp(-mu*h)
            # continuous condition of du
            index[4*N+3*N*(N-1)+N*j+i] = np.array([4*N*j+4*i,4*N*j+4*i+1,4*N*j+4*i+2,4*N*j+4*i+3,4*N*(j+1)+4*i,4*N*(j+1)+4*i+1,4*N*(j+1)+4*i+2,4*N*(j+1)+4*i+3])
            val[4*N+3*N*(N-1)+N*j+i] = np.array([0,0,mu0*np.exp((mu0-mu)*h),-mu0*np.exp(-(mu0+mu)*h),0,0,-mu1*np.exp(-(mu1+mu)*h),mu1*np.exp((mu1-mu)*h)])
            B[4*N+3*N*(N-1)+N*j+i] = 0
            #interface
            if (i, j) in buttom:
                B[4*N+2*N*(N-1)+N*j+i] = B[4*N+2*N*(N-1)+N*j+i] - alpha*np.exp(-mu*h)
                B[4*N+3*N*(N-1)+N*j+i] = B[4*N+3*N*(N-1)+N*j+i] - beta*np.exp(-mu*h)
                index_jump.extend([4*N+2*N*(N-1)+N*j+i, 4*N+3*N*(N-1)+N*j+i])
            elif (i, j) in top:
                B[4*N+2*N*(N-1)+N*j+i] = B[4*N+2*N*(N-1)+N*j+i] + alpha*np.exp(-mu*h)
                B[4*N+3*N*(N-1)+N*j+i] = B[4*N+3*N*(N-1)+N*j+i] + beta*np.exp(-mu*h)
                index_jump.extend([4*N+2*N*(N-1)+N*j+i, 4*N+3*N*(N-1)+N*j+i])
    
    #计算解
    U_data = val.flatten()
    U_row = np.repeat(np.arange(4*N**2), 8)
    U_col = index.flatten()
    U_sparse = sparse.csr_matrix((U_data, (U_row, U_col)), shape=(4*N**2, 4*N**2))
    C = linalg.spsolve(U_sparse, B)
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
    index_boundary = [i for i in range(4*N)]
    index_continuous = list(set([i for i in range(4*N, 4*N**2)])-set(index_jump))
    return B, C, up, index, val, index_boundary, index_jump, index_continuous

def generate_interface(N):
    vertical_left = set([(N//4-1, j) for j in range(N//4, N*3//4)])
    vertical_right = set([(N//2-1, j) for j in range(N//2, N*3//4)])|set([(N*3//4-1, j) for j in range(N//4, N//2)])
    horizontal_buttom = set([(i, N//4-1) for i in range(N//4, N*3//4)])
    horizontal_top = set([(i, N//2-1) for i in range(N//2, N*3//4)])|set([(i, N*3//4-1) for i in range(N//4, N//2)])
    interface = {'vertical_left': vertical_left, 'vertical_right': vertical_right, 
                 'horizontal_buttom': horizontal_buttom, 'horizontal_top': horizontal_top}
    return interface

if __name__ == '__main__':
    N = 32
    ntrain = 1000
    ntest = 200
    ntotal = ntrain + ntest
    alpha = 1.0 # interface jump
    beta = 0
    eps = 1.0  # We multiply both sides of the equation by 1/eps, so eps here can be 1.0
    # f = generate(samples = ntotal, out_dim=N+1, length_scale=1)
    # f *= 1000.0
    # np.save(r'DeepONet-type\2d-L-shaped\saved_data\f.npy', f)
    f = np.load(r'DeepONet-type\2d-L-shaped\saved_data\f.npy')

    k = 0 
    x = np.linspace(0, 1, N+1)
    xx,yy = np.meshgrid(x,x)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xx, yy, f[k], cmap='rainbow')
    ax.title.set_text('generated f(x,y)')
    plt.show()

    B_total = np.zeros((ntotal,4*N**2), dtype=np.float32)
    C_total = np.zeros((ntotal,4*N**2), dtype=np.float32)
    up_total = np.zeros((ntotal,N,N), dtype=np.float32)
    f_total = np.zeros((ntotal,(N+1)**2), dtype=np.float32)
    interface = generate_interface(N)
    for k in range(ntotal):
        B, C, up, index, val, index_b, index_j, index_c = tfpm2d(N, f[k], interface)
        B_total[k] = B
        C_total[k] = C
        up_total[k] = up
        f_total[k] = f[k].reshape(-1)

    # index of L-shaped(centor of cell)
    # idx_y = [[N//4+i]*(N//2) for i in range(0, N//4)] + [[N//2+i]*(N//4) for i in range(0, N//4)]
    # idx_y_remain = [[i]*N for i in range(0, N//4)] + [[N//4+i]*(N//2) for i in range(0, N//4)]\
    #                 + [[N//2+i]*(N*3//4) for i in range(0, N//4)] + [[N*3//4+i]*N for i in range(0, N//4)]
    # idx_y = np.concatenate(idx_y)
    # idx_y_remain = np.concatenate(idx_y_remain)
    # idx_x = [N//4+j for j in range(N // 2)]*(N//4) + [N//4+j for j in range(N//4)]*(N//4)
    # idx_x_remain = np.hstack((np.array([j for j in range(N)]*(N//4)),
    #                         np.concatenate([[j for j in range(N//4)]+[j for j in range(N*3//4, N)]]*(N//4)), 
    #                         np.concatenate([[j for j in range(N//4)]+[j for j in range(N//2, N)]]*(N//4)),
    #                         np.array([j for j in range(N)]*(N//4))))
    # idx_x = np.array(idx_x)

    # x = np.linspace(1/2/N, 1-1/2/N, N)
    # xx, yy = np.meshgrid(x, x)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # # plot inner L-shaped area
    # Z_full = np.full_like(xx, np.nan, dtype=float)
    # Z_full[idx_y, idx_x] = up_total[0, idx_y, idx_x]
    # ax.plot_surface(xx, yy, Z_full, cmap='rainbow')
    # # plot remaining area
    # Z_full = np.full_like(xx, np.nan, dtype=float)
    # Z_full[idx_y_remain, idx_x_remain] = up_total[0, idx_y_remain, idx_x_remain]
    # ax.plot_surface(xx, yy, Z_full, cmap='rainbow')
    # ax.title.set_text('generated f(x,y)')
    # plt.show()

    M = 4 # M-times test-resolution
    ut_train_vertex = np.zeros((ntrain, N+1, N+1))
    ut_test_fine = np.zeros((ntest, M*N+1, M*N+1))
    grid_fine = np.linspace(0,1,N*M+1)
    X, Y = np.meshgrid(grid_fine, grid_fine)
    points = np.stack((Y.flatten(), X.flatten()), axis=-1)
    f_test_fine = np.zeros((ntest, M*N+1, M*N+1))
    interface = generate_interface(N*M)
    import test_tfpm_refine

    for k in range(ntotal):
        interpolate_f_2d = interpolate.RegularGridInterpolator((np.linspace(0, 1, N+1),np.linspace(0, 1, N+1)), f[k])
        f_fine_each = interpolate_f_2d(points).reshape(N*M+1, N*M+1)
        _, _, ut, _, _ = test_tfpm_refine.tfpm2d(N*M, f_fine_each, interface)
        if k<ntrain:
            ut_train_vertex[k] = ut[::M, ::M]
        else:
            ut_test_fine[k-ntrain] = ut[:]
            f_test_fine[k-ntrain] = f_fine_each[:]
    np.savez(r"DeepONet-type\2d-L-shaped\saved_data\data.npz", f_total=f_total, B_total=B_total, C_total=C_total, up_total=up_total, index=index, val=val, u_test_fine=ut_test_fine, index_continuous=index_c, index_jump=index_j, index_boundary=index_b)
    np.savez(r"DeepONet-type\2d-L-shaped\saved_data\data_fno.npz", u_test_fine=ut_test_fine, u_train_sparse=ut_train_vertex, f_test_fine=f_test_fine)