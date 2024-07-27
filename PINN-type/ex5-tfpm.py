

import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from math import sqrt

def F(x,y):
    # return 0
    return np.sin(np.pi*(x+y))

def c(x,y):
    if x < 1/2:
        a = 4
    else:
        a = 1
    return 1000*4*a #坐标变换
    #return 1+x+y

def b(x,y):
    if x >= 1/2:
        a = 2*(1-x)
    else:
        a = 0
    return a
    #return 0

alpha = 1 #interface jump
beta = 0
eps = 1.0    
N = 32
h = 1/(2*N)
U = np.zeros((4*N**2,4*N**2))
B = np.zeros(4*N**2)

#boundary y=0.
for i in range(0,N):
    x0 = (2*i+1)*h
    y0 = h
    f0 = F(x0,y0)
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
up0 = np.zeros((N+2,N+2))
up0[1:N+1,1:N+1] = up
for k in range(0,N+2):
    s = k/(N+1)
    up0[0,k] = b(s,0)
    up0[N+1,k] = b(s,1)
    up0[k,0] = b(0,s)
    up0[k,N+1] = b(1,s)
x = np.linspace(h,1-h,N)
x0 = np.zeros(N+2)
x0[1:N+1] = x
# xx,yy = np.meshgrid(x0,x0)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(xx, yy, up0, cmap='rainbow')
# plt.show()

#refinement
M = 10
u_refine = np.zeros((M*N+1,M*N+1))
hh = 1/(M*N)
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
        for ki in range(0,M+1):
            for kj in range(0,M+1):
                xhi = -h + ki*hh
                xhj = -h + kj*hh
                # if np.exp(mu0*xhi)>1e10 or np.exp(-mu0*xhi)>1e3 or np.exp(mu0*xhj)>1e3 or np.exp(-mu0*xhj)>1e3: print(max(np.exp(mu0*xhi), np.exp(mu0*xhj)))
                # print('c',c1, c2,c3,c4)
                # print('m',c1*np.exp(mu0*xhi), c2*np.exp(-mu0*xhi), c3*np.exp(mu0*xhj), c4*np.exp(-mu0*xhj))
                c11 = 0.0 if c1<np.exp(-21-mu0*xhi) else c1*np.exp(mu0*xhi)
                c22 = 0.0 if c2<np.exp(-21+mu0*xhi) else c2*np.exp(-mu0*xhi)
                c33 = 0.0 if c3<np.exp(-21-mu0*xhj) else c3*np.exp(mu0*xhj)
                c44 = 0.0 if c4<np.exp(-21+mu0*xhj) else c4*np.exp(-mu0*xhj)
                u_refine[j*M+kj,i*M+ki] = f0/c0 + c11 + c22 + c33 + c44 
for k in range(0,M*N+1):
    s = k*hh
    u_refine[0,k] = b(s,0)
    u_refine[M*N,k] = b(s,1)
    u_refine[k,0] = b(0,s)
    u_refine[k,M*N] = b(1,s)
xh = np.linspace(0,1,N*M+1)
yh = np.linspace(0,1,N*M+1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xxh, yyh = np.meshgrid(xh[:int(N*M/2)], yh)
ax.plot_surface(xxh, yyh, u_refine[:, :int(N*M/2)], cmap='rainbow')
xxh, yyh = np.meshgrid(xh[int(N*M/2):], yh)
ax.plot_surface(xxh, yyh, u_refine[:, int(N*M/2):], cmap='rainbow')
plt.show()


