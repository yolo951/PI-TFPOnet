
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from math import sqrt


def TFPM_get_discrete_u(N, p, q, b, f, eps):
    num_eq = (N-1)*(N-1)
    
    h = 1/N
    mu = np.sqrt(b + (p**2 + q**2)/4/eps**2)/eps
    coeff_u1 = np.exp(-p*h/2/eps**2)/4/np.cosh(mu*h/2)**2
    coeff_u2 = np.exp(-q*h/2/eps**2)/4/np.cosh(mu*h/2)**2
    coeff_u3 = np.exp(p*h/2/eps**2)/4/np.cosh(mu*h/2)**2
    coeff_u4 = np.exp(q*h/2/eps**2)/4/np.cosh(mu*h/2)**2
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

        i = int(x // h)
        j = int(y // h)
        
        if i >= N: i = N-1
        if j >= N: j = N-1
        
        a0_ij = self.a0[i, j]
        mu0_ij = self.mu0[i, j]
        
        u1, u2, u3, u4 = self.u[i, j], self.u[i, j+1], self.u[i+1, j+1], self.u[i+1, j]
        
        sqrt2 = sqrt(2)
        coeff = np.exp((self.p0[i, j]*x+self.q0[i, j]*y)/2/eps**2)
        basis1 = coeff * np.exp(mu0_ij * (x + y) / sqrt2)
        basis2 = coeff * np.exp(-mu0_ij * (x + y) / sqrt2)
        basis3 = coeff * np.exp(mu0_ij * (x - y) / sqrt2)
        basis4 = coeff * np.exp(-mu0_ij * (x - y) / sqrt2)
        
        muh = mu0_ij*h/sqrt2
        # A = 1/(1+np.exp(-4*muh)-2*np.exp(-2*muh)) * np.array([
        #     [np.exp(-3*muh), -np.exp(-2*muh), np.exp(-muh), -np.exp(-2*muh)],
        #     [np.exp(-muh), -np.exp(-2*muh), np.exp(-3*muh), -np.exp(-2*muh)],
        #     [-np.exp(-2*muh), np.exp(-muh), -np.exp(-2*muh), np.exp(-3*muh)],
        #     [-np.exp(-2*muh), np.exp(-3*muh), -np.exp(-2*muh), np.exp(-muh)]
        # ])
        # b1 = (u1-a0_ij)*np.exp((self.p0[i, j]+self.q0[i, j])*h/4/eps**2)
        # b2 = (u2-a0_ij)*np.exp((self.q0[i, j]-self.p0[i, j])*h/4/eps**2)
        # b3 = (u3-a0_ij)*np.exp((-self.p0[i, j]-self.q0[i, j])*h/4/eps**2)
        # b4 = (u4-a0_ij)*np.exp((self.p0[i, j]-self.q0[i, j])*h/4/eps**2)

        A_prime = 1/4/np.sinh(muh)**2 * np.array([
            [np.exp(-muh), -1.0, np.exp(muh), -1.0],
            [np.exp(muh), -1.0, np.exp(-muh), -1.0],
            [-1.0, np.exp(muh), -1.0, np.exp(-muh)],
            [-1.0, np.exp(-muh), -1.0, np.exp(muh)]
        ])
        # b1 = (u1-a0_ij)*np.exp((self.p0[i, j]+self.q0[i, j])*h/4/eps**2)
        # b2 = (u2-a0_ij)*np.exp((self.q0[i, j]-self.p0[i, j])*h/4/eps**2)
        # b3 = (u3-a0_ij)*np.exp((-self.p0[i, j]-self.q0[i, j])*h/4/eps**2)
        # b4 = (u4-a0_ij)*np.exp((self.p0[i, j]-self.q0[i, j])*h/4/eps**2)
        # B = np.array([b1, b2, b3, b4])
        
        x0, y0 = self.x[i], self.y[j]
        A = np.array([
            [np.exp((self.p0[i, j]+self.q0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*x0+self.q0[i,j]*y0)/2/eps**2)*np.exp(mu0_ij*(x0+y0)/sqrt2),
             np.exp((self.p0[i, j]+self.q0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*x0+self.q0[i,j]*y0)/2/eps**2)*np.exp(-mu0_ij*(x0+y0)/sqrt2),
             np.exp((self.p0[i, j]+self.q0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*x0+self.q0[i,j]*y0)/2/eps**2)*np.exp(mu0_ij*(x0-y0)/sqrt2),
             np.exp((self.p0[i, j]+self.q0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*x0+self.q0[i,j]*y0)/2/eps**2)*np.exp(mu0_ij*(y0-x0)/sqrt2)],
            [np.exp((self.q0[i, j]-self.p0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*(x0+h)+self.q0[i,j]*y0)/2/eps**2)*np.exp(mu0_ij*(x0+y0+h)/sqrt2),
             np.exp((self.q0[i, j]-self.p0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*(x0+h)+self.q0[i,j]*y0)/2/eps**2)*np.exp(-mu0_ij*(x0+y0+h)/sqrt2),
             np.exp((self.q0[i, j]-self.p0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*(x0+h)+self.q0[i,j]*y0)/2/eps**2)*np.exp(mu0_ij*(x0+h-y0)/sqrt2),
             np.exp((self.q0[i, j]-self.p0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*(x0+h)+self.q0[i,j]*y0)/2/eps**2)*np.exp(mu0_ij*(y0-x0-h)/sqrt2)],
            [np.exp((-self.p0[i, j]-self.q0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*(x0+h)+self.q0[i,j]*(y0+h))/2/eps**2)*np.exp(mu0_ij*(x0+y0+2*h)/sqrt2),
             np.exp((-self.p0[i, j]-self.q0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*(x0+h)+self.q0[i,j]*(y0+h))/2/eps**2)*np.exp(-mu0_ij*(x0+y0+2*h)/sqrt2),
             np.exp((-self.p0[i, j]-self.q0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*(x0+h)+self.q0[i,j]*(y0+h))/2/eps**2)*np.exp(mu0_ij*(x0-y0)/sqrt2),
             np.exp((-self.p0[i, j]-self.q0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*(x0+h)+self.q0[i,j]*(y0+h))/2/eps**2)*np.exp(mu0_ij*(y0-x0)/sqrt2)],
            [np.exp((self.p0[i, j]-self.q0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*x0+self.q0[i,j]*(y0+h))/2/eps**2)*np.exp(mu0_ij*(x0+y0+h)/sqrt2),
             np.exp((self.p0[i, j]-self.q0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*x0+self.q0[i,j]*(y0+h))/2/eps**2)*np.exp(-mu0_ij*(x0+y0+h)/sqrt2),
             np.exp((self.p0[i, j]-self.q0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*x0+self.q0[i,j]*(y0+h))/2/eps**2)*np.exp(mu0_ij*(x0-y0-h)/sqrt2),
             np.exp((self.p0[i, j]-self.q0[i, j])*h/4/eps**2)*np.exp((self.p0[i,j]*x0+self.q0[i,j]*(y0+h))/2/eps**2)*np.exp(mu0_ij*(y0+h-x0)/sqrt2)]
        ])
        b1 = (u1-a0_ij)*np.exp((self.p0[i, j]+self.q0[i, j])*h/4/eps**2)
        b2 = (u2-a0_ij)*np.exp((self.q0[i, j]-self.p0[i, j])*h/4/eps**2)
        b3 = (u3-a0_ij)*np.exp((-self.p0[i, j]-self.q0[i, j])*h/4/eps**2)
        b4 = (u4-a0_ij)*np.exp((self.p0[i, j]-self.q0[i, j])*h/4/eps**2)
        B = np.array([b1, b2, b3, b4])
        c = np.linalg.solve(A, B)
        print(A*A_prime)
        # c = np.matmul(A, B)
        
        u_h_val = a0_ij + c[0] * basis1 + c[1] * basis2 + c[2] * basis3 + c[3] * basis4
        
        return u_h_val
    
    def recovery(self, x, y):

        assert x.shape == y.shape, "Shapes of x and y must match"
        vectorized_u_h = np.vectorize(self.recovery_each)
        return vectorized_u_h(x, y)

N = 12
x = np.linspace(0, 1, N+1)
y = np.linspace(0, 1, N+1)
xx, yy = np.meshgrid(x, y)
eps = 1.0
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
tfpm2d = TFPM2d(u_approx, N, p, q, b, f, eps)
u_recovery = tfpm2d.recovery(xx, yy)
# print(np.max(np.abs(u_approx-u_exact)))
# print(np.max(np.abs(u_recovery-u_exact)))
# print(u_recovery-u_approx)


fig = plt.figure()
axs1 = fig.add_subplot(121, projection='3d')
axs1.plot_surface(xx, yy, u_recovery, cmap='rainbow', label='approximation')
axs1.view_init(elev=-10., azim=30)
axs2 = fig.add_subplot(122, projection='3d')
axs2.plot_surface(xx, yy, u_exact, cmap='rainbow', label='exact')
axs2.view_init(elev=-10., azim=30)
plt.show()
