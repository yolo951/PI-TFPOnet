
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from math import sqrt


def TFPM_get_discrete_u(N, p, q, b, f, eps):
    num_eq = (N-1)*(N-1)
    
    h = 1/N
    mu = np.sqrt(b + (p**2 + q**2)/4/eps**2)/eps
    coeff_u1 = np.exp(-(p/2/eps**2+mu)*h)/(1+np.exp(-2*mu*h)+2*np.exp(-mu*h))
    coeff_u2 = np.exp(-(q/2/eps**2+mu)*h)/(1+np.exp(-2*mu*h)+2*np.exp(-mu*h))
    coeff_u3 = np.exp((p/2/eps**2-mu)*h)/(1+np.exp(-2*mu*h)+2*np.exp(-mu*h))
    coeff_u4 = np.exp((q/2/eps**2-mu)*h)/(1+np.exp(-2*mu*h)+2*np.exp(-mu*h))
    rhs = f/b*(1-coeff_u1-coeff_u2-coeff_u3-coeff_u4)
    index = lambda i, j: (i - 1) * (N - 1) + j - 1
    A = np.zeros((num_eq, num_eq))
    b_vec = np.zeros(num_eq)
    u = np.zeros((N+1, N+1))

    eq = 0
    for i in range(1, N):
        for j in range(1, N):
            u0_index = index(i, j)
            u1_index = index(i, j+1)
            u2_index = index(i+1, j)
            u3_index = index(i, j-1)
            u4_index = index(i-1, j)

            A[eq, u0_index] = 1
            if j<N-1: A[eq, u1_index] = -coeff_u1[i, j]
            if i<N-1: A[eq, u2_index] = -coeff_u2[i, j]
            if j>1: A[eq, u3_index] = -coeff_u3[i, j]
            if i>1: A[eq, u4_index] = -coeff_u4[i, j]
            b_vec[eq] = rhs[i, j]
            
            eq += 1
    u[1:-1, 1:-1] = np.linalg.solve(A, b_vec).reshape((N-1, N-1))
    return u

class TFPM2d:
    def __init__(self, u, N, p, q, b, f, eps):
        self.u = u
        self.N = N
        self.p = p
        self.q = q
        self.b = b
        self.f = f
        self.eps = eps

        
        self.x = np.linspace(0, 1, N+1)
        self.y = np.linspace(0, 1, N+1)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        

        self.a0 = np.zeros((N, N))
        self.mu0 = np.zeros((N, N))
        self.p0 = np.zeros((N, N))
        self.q0 = np.zeros((N, N))
        
        self.area = (1/N) * (1/N)
        
        self.get_integrals()
        
    
    def get_integrals(self):
        for i in range(self.N):
            for j in range(self.N):
                x0, x1 = self.x[i], self.x[i+1]
                y0, y1 = self.y[j], self.y[j+1]
                
                f0, _ = dblquad(f, x0, x1, lambda x: y0, lambda x: y1)
                f0 /= self.area
                b0, _ = dblquad(b, x0, x1, lambda x: y0, lambda x: y1)
                b0 /= self.area
                self.a0[i, j] = f0/b0
                self.p0[i, j], _ = dblquad(p, x0, x1, lambda x: y0, lambda x: y1)
                self.q0[i, j], _ = dblquad(q, x0, x1, lambda x: y0, lambda x: y1)
                self.p0[i, j] /= self.area
                self.q0[i, j] /= self.area
                self.mu0[i, j] = 1/eps*np.sqrt(b0 + (self.p0[i, j]**2 + self.q0[i, j]**2)/4/eps**2)

    def recovery_each(self, x, y):
        h = 1 / N

        j = int(x // h)
        i = int(y // h)
        
        if i >= N: i = N-1
        if j >= N: j = N-1
        
        a0_ij = self.a0[i, j]
        mu0_ij = self.mu0[i, j]
        
        u1, u2, u3, u4 = self.u[i, j], self.u[i, j+1], self.u[i+1, j+1], self.u[i+1, j]
        # x0, y0 = self.x[i], self.y[j]
        x, y = x - self.x[j] - h/2, y - self.y[i] - h/2
        
        sqrt2 = sqrt(2)
        
        a11 = (self.p0[i, j]+self.q0[i, j])*h/4/eps**2 - mu0_ij*h/sqrt2 - sqrt2*mu0_ij*h + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 + mu0_ij * (x + y) / sqrt2
        a12 = (self.q0[i, j]-self.p0[i, j])*h/4/eps**2 - sqrt2*mu0_ij*h  + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 + mu0_ij * (x + y) / sqrt2
        a13 = (-self.p0[i, j]-self.q0[i, j])*h/4/eps**2 + mu0_ij*h/sqrt2 - sqrt2*mu0_ij*h + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 + mu0_ij * (x + y) / sqrt2
        a14 = (self.p0[i, j]-self.q0[i, j])*h/4/eps**2 - sqrt2*mu0_ij*h  + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 + mu0_ij * (x + y) / sqrt2
        #c1 = (u1-a0_ij)*np.exp(a11) - (u2-a0_ij)*np.exp(a12) + (u3-a0_ij)*np.exp(a13)  - (u4-a0_ij)*np.exp(a14)
        #c1 = c1 / (1 + np.exp(-2*sqrt2*mu0_ij*h) - 2*np.exp(-sqrt2*mu0_ij*h))
        
        a21 = (self.p0[i, j]+self.q0[i, j])*h/4/eps**2 + mu0_ij*h/sqrt2 - sqrt2*mu0_ij*h + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 - mu0_ij * (x + y) / sqrt2
        a22 = (self.q0[i, j]-self.p0[i, j])*h/4/eps**2 - sqrt2*mu0_ij*h  + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 - mu0_ij * (x + y) / sqrt2
        a23 = (-self.p0[i, j]-self.q0[i, j])*h/4/eps**2 - mu0_ij*h/sqrt2 - sqrt2*mu0_ij*h + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 - mu0_ij * (x + y) / sqrt2
        a24 = (self.p0[i, j]-self.q0[i, j])*h/4/eps**2 - sqrt2*mu0_ij*h  + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 - mu0_ij * (x + y) / sqrt2
        #c2 = (u1-a0_ij)*np.exp(a21) - (u2-a0_ij)*np.exp(a22) + (u3-a0_ij)*np.exp(a23) - (u4-a0_ij)*np.exp(a24)
        #c2 = c2 / (1 + np.exp(-2*sqrt2*mu0_ij*h) - 2*np.exp(-sqrt2*mu0_ij*h))
        
        a31 = (self.p0[i, j]+self.q0[i, j])*h/4/eps**2 - sqrt2*mu0_ij*h + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 + mu0_ij * (x - y) / sqrt2
        a32 = (self.q0[i, j]-self.p0[i, j])*h/4/eps**2 + mu0_ij*h/sqrt2 - sqrt2*mu0_ij*h  + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 + mu0_ij * (x - y) / sqrt2
        a33 = (-self.p0[i, j]-self.q0[i, j])*h/4/eps**2 - sqrt2*mu0_ij*h + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 + mu0_ij * (x - y) / sqrt2
        a34 = (self.p0[i, j]-self.q0[i, j])*h/4/eps**2 - mu0_ij*h/sqrt2 - sqrt2*mu0_ij*h  + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 + mu0_ij * (x - y) / sqrt2
        #c3 = -(u1-a0_ij)*np.exp(a31) + (u2-a0_ij)*np.exp(a32) - (u3-a0_ij)*np.exp(a33)  + (u4-a0_ij)*np.exp(a34)
        #c3 = c3 / (1 + np.exp(-2*sqrt2*mu0_ij*h) - 2*np.exp(-sqrt2*mu0_ij*h))
        
        a41 = (self.p0[i, j]+self.q0[i, j])*h/4/eps**2 - sqrt2*mu0_ij*h + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 + mu0_ij * (y - x) / sqrt2
        a42 = (self.q0[i, j]-self.p0[i, j])*h/4/eps**2 - mu0_ij*h/sqrt2 - sqrt2*mu0_ij*h  + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 + mu0_ij * (y - x) / sqrt2
        a43 = (-self.p0[i, j]-self.q0[i, j])*h/4/eps**2 - sqrt2*mu0_ij*h + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 + mu0_ij * (y - x) / sqrt2
        a44 = (self.p0[i, j]-self.q0[i, j])*h/4/eps**2 + mu0_ij*h/sqrt2 - sqrt2*mu0_ij*h  + (self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2 + mu0_ij * (y - x) / sqrt2
        #c4 = -(u1-a0_ij)*np.exp(a41) + (u2-a0_ij)*np.exp(a42) - (u3-a0_ij)*np.exp(a43)  + (u4-a0_ij)*np.exp(a44)
        #c4 = c4 / (1 + np.exp(-2*sqrt2*mu0_ij*h) - 2*np.exp(-sqrt2*mu0_ij*h))
        
        if (a11>10)|(a12>10)|(a13>10)|(a14>10)|(a21>10)|(a22>10)|(a23>10)|(a24>10)|(a31>10)|(a32>10)|(a33>10)|(a34>10)|(a41>10)|(a42>10)|(a43>10)|(a44>10):
            print('large number to nan.')
            print('one',a11,a12,a13,a14)
            print('two',a21,a22,a23,a24)
            print('three',a31,a32,a33,a34)
            print('four',a41,a42,a43,a44)
        c_sum = (u1-a0_ij)*(np.exp(a11)+np.exp(a21)-np.exp(a31)-np.exp(a41)) + (u2-a0_ij)*(-np.exp(a12)-np.exp(a22)+np.exp(a32)+np.exp(a42)) + (u3-a0_ij)*(np.exp(a13)+np.exp(a23)-np.exp(a33)-np.exp(a43)) + (u4-a0_ij)*(-np.exp(a14)-np.exp(a24)+np.exp(a34)+np.exp(a44))
        c_sum = c_sum / (1 + np.exp(-2*sqrt2*mu0_ij*h) - 2*np.exp(-sqrt2*mu0_ij*h))
        
        
        u_h_val = a0_ij + c_sum
        
        return u_h_val
    
    def recovery(self, x, y):

        assert x.shape == y.shape, "Shapes of x and y must match"
        vectorized_u_h = np.vectorize(self.recovery_each)
        return vectorized_u_h(x, y)

N = 3
x = np.linspace(0, 1, N+1)
y = np.linspace(0, 1, N+1)
xx, yy = np.meshgrid(x, y)
eps = sqrt(0.001)
p = np.vectorize(lambda x, y: 1.0)
q = np.vectorize(lambda x, y: 0.0)
b = np.vectorize(lambda x, y: 1.0)
f = np.vectorize(lambda x, y: (2*eps**2+y*(1-y))*(np.exp((x-1)/eps**2)+(x-1)*np.exp(-1/eps**2)-x)+y*(1-y)*(np.exp(-1/eps**2)-1))
p_val = p(xx, yy)
q_val = q(xx, yy)
b_val = b(xx, yy)
f_val = f(xx, yy)

u_approx = TFPM_get_discrete_u(N, p_val, q_val, b_val, f_val, eps)
u_exact = yy*(1-yy)*(np.exp((xx-1)/eps**2)+(xx-1)*np.exp(-1/eps**2)-xx)

print(np.max(np.abs(u_approx-u_exact)))

Nh = 4
xh = np.linspace(0, 1, Nh+1)
yh = np.linspace(0, 1, Nh+1)
xxh, yyh = np.meshgrid(xh, yh)
u_exact_h = yyh*(1-yyh)*(np.exp((xxh-1)/eps**2)+(xxh-1)*np.exp(-1/eps**2)-xxh)
tfpm2d = TFPM2d(u_approx, N, p, q, b, f, eps)
u_recovery = tfpm2d.recovery(xxh, yyh)
print(np.max(np.abs(u_recovery-u_exact_h)))



fig = plt.figure()
axs1 = fig.add_subplot(121, projection='3d')
axs1.plot_surface(xxh, yyh, u_recovery, cmap='rainbow', label='approximation')
axs1.view_init(elev=-10., azim=30)
axs2 = fig.add_subplot(122, projection='3d')
axs2.plot_surface(xx, yy, u_exact, cmap='rainbow', label='exact')
axs2.view_init(elev=-10., azim=30)
plt.show()
