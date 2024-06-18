
import torch
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy
from scipy.integrate import quad
from math import sqrt
from scipy import interpolate
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# solve equation -u''(x)+q(x)u(x)=F(x)
def discontinuous_tfpm(f, N):

    def q(x):
        return np.where(x<=0.5, 5.0, 0.1*(4+32*x))

    def q2(x):
        return 0.1*(4+32*x)

    def F(x):
        return f(x)
        # return interpolate_f(x)[k]

    def integrand_linear(s):  # for -u''=f
        G = np.where(y >= s, s-y, 0)
        return F(s)*G

    def integrandp_linear(s):  # for -u''=f
        G = np.where(y >= s, -1.0, 0)
        return F(s)*G

    def integrand_sinh(s):  # for -u''+bu = f, b>0
        G = 1/2/sqrt(b)*np.where(y >= s, np.sinh(sqrt(b)*(s-y)), np.sinh(sqrt(b)*(y-s)))
        return F(s) * G

    def integrandp_sinh(s):
        G = 1/2*np.where(y > s, -np.cosh(sqrt(b)*(s-y)), np.where(y<s, np.cosh(sqrt(b)*(y-s)), 0.0))
        return F(s) * G

    def integrand_airy(s):  # for -u'' + (ax+b)u = f
        G = np.pi/2*np.where(z >= s, airy(s)[2] * airy(z)[0] - airy(s)[0] * airy(z)[2],
                              airy(s)[0] * airy(z)[2] - airy(s)[2] * airy(z)[0])
        return np.power(np.abs(a), -2/3)*G*F(s/np.cbrt(a)-b/a)

    def integrandp_airy(s):
        G = np.pi/2 * np.where(z > s, airy(s)[2] * airy(z)[1] - airy(s)[0] * airy(z)[3],
                               np.where(z<s, airy(s)[0] * airy(z)[3] - airy(s)[2] * airy(z)[1], 0.0))
        return 1/np.cbrt(a)*G*F(s/np.cbrt(a)-b/a)

    # interpolate_f = interpolate.interp1d(np.linspace(0, 1, N), f)
    x = np.zeros((N), dtype=np.float64)
    x1 = np.zeros((int((N+1)/2)), dtype=np.float64)
    x2 = np.zeros((int((N + 1) / 2)), dtype=np.float64)
    U = np.zeros((2*(N-1), 2*(N-1)), dtype=np.float64)
    B = np.zeros((2*(N-1),), dtype=np.float64)

    grid = np.linspace(0, 1, N)
    cRight = q(grid)
    cLeft = q(grid)
    cRight[int((N-1)/2)] = q2(grid[int((N-1)/2)])
    # interp_coeff = get_coeff(grid)
    c = np.polyfit(grid[:2], np.array([cRight[0], cLeft[1]]), 1)
    a, b = c[0], c[1]
    # a, b = interp_coeff[0]
    if abs(a) < 1e-8:
        if abs(b) < 1e-8:
            lambda1 = 1.0
            mu1 = grid[0]
            y = grid[0]
            F1 = quad(integrand_linear, grid[0], grid[1])[0]
            lambda1p = 0.0
            mu1p = 1.0
            lambda1pp = 0.0
            mu1pp = 0.0
        else:
            lambda1 = np.exp(grid[0] * sqrt(b))
            mu1 = np.exp(-grid[0] * sqrt(b))
            y = grid[0]
            F1 = quad(integrand_sinh, grid[0], grid[1])[0]
            lambda1p = sqrt(b) * np.exp(grid[0] * sqrt(b))
            mu1p = -sqrt(b) * np.exp(-grid[0] * sqrt(b))
            lambda1pp = b * np.exp(grid[0] * sqrt(b))
            mu1pp = b * np.exp(-grid[0] * sqrt(b))
    else:
        z1 = q(grid[0]) * np.power(np.abs(a), -2 / 3)  # 如果直接使用np.power(a, -2/3),可能会计算出来复数
        z2 = q(grid[1]) * np.power(np.abs(a), -2 / 3)
        lambda1 = airy(z1)[0]
        mu1 = airy(z1)[2]
        z = z1
        F1 = quad(integrand_airy, z1, z2)[0] if z2 >= z1 else quad(integrand_airy, z2, z1)[0]
        lambda1p = np.cbrt(a)*airy(z1)[1]
        mu1p = np.cbrt(a)*airy(z1)[3]
        lambda1pp = z1 * airy(z1)[0]*np.cbrt(a)**2
        mu1pp = z1 * airy(z1)[2]*np.cbrt(a)**2
    U[0,:2] = np.array([-lambda1, -mu1])
    B[0] = F1
    coeff = [[lambda1, mu1, F1]]
    coeffp = [[lambda1p, mu1p]]
    coeffpp = [[lambda1pp, mu1pp]]
    for i in range(1, 2*N-3, 2):
        c= np.polyfit(grid[i//2:i//2+2], np.array([cRight[i//2], cLeft[i//2+1]]), 1)
        a1, b1 = c[0], c[1]
        c = np.polyfit(grid[i//2+1:i//2+3], np.array([cRight[i//2+1], cLeft[i//2+2]]), 1)
        a2, b2 = c[0], c[1]
        # a1, b1 = interp_coeff[i // 2]
        # a2, b2 = interp_coeff[i // 2 + 1]
        if abs(a1) < 1e-8:
            if abs(b1) < 1e-8:
                lambda1, gamma1, delta1 = 1.0, 0.0, 1.0
                mu1 = grid[i//2+1]
                y = grid[i//2 + 1]
                F1 = quad(integrand_linear, grid[i//2], grid[i//2+1])[0]
                G1 = -quad(F, grid[i//2], grid[i//2+1])[0]
                lambda1pp = 0.0
                mu1pp = 0.0
            else:
                # b1 = abs(b1)
                # b2 = abs(b2)
                lambda1 = np.exp(sqrt(b1) * grid[i // 2 + 1])
                gamma1 = sqrt(b1) * lambda1
                mu1 = np.exp(-sqrt(b1) * grid[i // 2 + 1])
                delta1 = -sqrt(b1) * mu1
                y, b = grid[i // 2 + 1], b1
                F1 = quad(integrand_sinh, grid[i // 2], grid[i // 2 + 1])[0]
                G1 = quad(integrandp_sinh, grid[i // 2], grid[i // 2 + 1])[0]
                lambda1pp = b1 * lambda1
                mu1pp = b1 * mu1
        else:
            z1 = cRight[i//2] * np.power(np.abs(a1), -2 / 3)
            z2 = cLeft[i//2+1]* np.power(np.abs(a1), -2 / 3)
            lambda1, gamma1, mu1, delta1 = airy(z2)
            gamma1 *= np.cbrt(a1)
            delta1 *= np.cbrt(a1)
            a, b, z = a1, b1, z2
            F1 = quad(integrand_airy, z1, z2)[0] if z2 >= z1 else quad(integrand_airy, z2, z1)[0]
            G1 = quad(integrandp_airy, z1, z2)[0] if z2 >= z1 else quad(integrandp_airy, z2, z1)[0]
            lambda1pp = z2 * lambda1*np.cbrt(a1)**2
            mu1pp = z2 * mu1*np.cbrt(a1)**2
        if abs(a2) < 1e-8:
            if abs(b2) < 1e-8:
                lambda2, gamma2, delta2 = 1, 0, 1
                mu2 = grid[i//2+1]
                y = grid[i//2 + 1]
                F2 = quad(integrand_linear, grid[i//2+1], grid[i//2+2])[0]
                G2 = 0.0
            else:
                lambda2 = np.exp(sqrt(b2) * grid[i // 2 + 1])
                gamma2 = sqrt(b2) * lambda2
                mu2 = np.exp(-sqrt(b2) * grid[i // 2 + 1])
                delta2 = -sqrt(b2) * mu2
                y, b = grid[i // 2 + 1], b2
                F2 = quad(integrand_sinh, grid[i // 2 + 1], grid[i // 2 + 2])[0]
                G2 = quad(integrandp_sinh, grid[i // 2 + 1], grid[i // 2 + 2])[0]
        else:
            z3 = cRight[i//2+1] * np.power(np.abs(a2), -2 / 3)
            z4 = cLeft[i//2+2] * np.power(np.abs(a2), -2 / 3)
            lambda2, gamma2, mu2, delta2 = airy(z3)
            gamma2 *= np.cbrt(a2)
            delta2 *= np.cbrt(a2)
            a, b, z = a2, b2, z3
            F2 = quad(integrand_airy, z3, z4)[0] if z4 >= z3 else quad(integrand_airy, z4, z3)[0]
            G2 = quad(integrandp_airy, z3, z4)[0] if z4 >= z3 else quad(integrandp_airy, z4, z3)[0]
        B[i] = F2 - F1
        B[i+1] = G2 - G1
        U[i, i-1:i+3] = np.array([lambda1, mu1, -lambda2, -mu2])
        U[i+1, i - 1:i + 3] = np.array([gamma1, delta1, -gamma2, -delta2])
        coeff.append([lambda1, mu1, F1])
        coeffp.append([gamma1, delta1])
        coeffpp.append([lambda1pp, mu1pp])
    c = np.polyfit(grid[-2:], np.array([cRight[-2], cLeft[-1]]), 1)
    a, b = c[0], c[1]
    # a, b = interp_coeff[-1]
    if abs(a)<1e-8:
        if abs(b)<1e-8:
            lambda1 = 1.0
            mu1 = grid[-1]
            y = grid[-1]
            F1 = quad(integrand_linear, grid[-2], grid[-1])[0]
            lambda1p = 0.0
            mu1p = 1.0
            lambda1pp = 0.0
            mu1pp = 0.0
        else:
            # b = abs(b)
            lambda1 = np.exp(sqrt(b)*grid[-1])
            mu1 = np.exp(-sqrt(b)*grid[-1])
            y = grid[-1]
            F1 = quad(integrand_sinh, grid[-2], grid[-1])[0]
            lambda1p = sqrt(b) * np.exp(grid[-1] * sqrt(b))
            mu1p = -sqrt(b) * np.exp(-grid[-1] * sqrt(b))
            lambda1pp = b * np.exp(grid[-1] * sqrt(b))
            mu1pp = b * np.exp(-grid[-1] * sqrt(b))
    else:
        z1 = q(grid[-2]) * np.power(np.abs(a), -2 / 3)
        z2 = q(grid[-1]) * np.power(np.abs(a), -2 / 3)
        lambda1 = airy(z2)[0]
        mu1 = airy(z2)[2]
        z = z2
        F1 = quad(integrand_airy, z1, z2)[0] if z2 >= z1 else quad(integrand_airy, z2, z1)[0]
        lambda1p = airy(z2)[1]*np.cbrt(a)
        mu1p = airy(z2)[3]*np.cbrt(a)
        lambda1pp = z2 * airy(z2)[0]*np.cbrt(a)**2
        mu1pp = z2 * airy(z2)[2]*np.cbrt(a)**2
    U[2*N-3, -2:] = np.array([-lambda1, -mu1])
    B[2*N-3] = F1
    B[N-2] -= 1.0
    B[N-1] -= 1.0
    coeff.append([lambda1, mu1, F1])
    coeffp.append([lambda1p, mu1p])
    coeffpp.append([lambda1pp, mu1pp])

    # M_inverse = np.diag(1 / np.max(np.abs(U), axis=0))
    # M_inverse = np.eye(U.shape[0])
    # scaled_U = np.matmul(U, M_inverse)
    AB = np.linalg.solve(U, B).flatten() # scaled AB

    # scale = np.diag(M_inverse)
    # scale1 = scale[::2]
    # scale2 = scale[1::2]
    # for i in range(len(coeffp)):
    #     j = i-1 if i>0 else 0
    #     coeff[i][0] *= scale1[j]
    #     coeff[i][1] *= scale2[j]
    #     coeffp[i][0] *= scale1[j]
    #     coeffp[i][1] *= scale2[j]
    #     coeffpp[i][0] *= scale1[j]
    #     coeffpp[i][1] *= scale2[j]

    # AB = np.linalg.solve(U, B).flatten()
    each_x = np.zeros(N)
    # boundary val = 0
    # each_x[0] = AB[0]*coeff[0][0]+AB[1]*coeff[0][1]+coeff[0][2]
    for i in range(1, N-1):
        each_x[i] = AB[2*(i-1)]*coeff[i][0]+AB[2*(i-1)+1]*coeff[i][1]+coeff[i][2]
    x[:] = each_x

    x1[:] = x[:int((N+1)/2)]
    x2[:] = x[int((N+1)/2)-1:]
    x2[0] += 1
    return x1, x2, U, B, coeff, AB, coeffp, coeffpp