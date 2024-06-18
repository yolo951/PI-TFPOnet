
import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.linalg import solve


def create_linear_system(N, eps, p, q, b, f):

    h = 1.0 / N  # Length of each small square
    mu = np.zeros((N, N))
    nu = np.zeros((N, N))
    # c = np.zeros((N, N))
    p_coeff = np.zeros((N, N))
    q_coeff = np.zeros((N, N))

    # u = nu + c*(a*exp(-mu*x) + b*exp(mu*x) + c*exp(-mu*y) + d*exp(mu*y))
    for i in range(N):
        for j in range(N):
            center_x = (j + 0.5) * h
            center_y = (i + 0.5) * h
            p0 = p(center_x, center_y)
            q0 = q(center_x, center_y)
            b0 = b(center_x, center_y)
            f0 = f(center_x, center_y)
            mu[i, j] = 1/eps*np.sqrt(b0+(p0**2+q0**2)/4/eps**2)
            nu[i, j] = f0/b0
            # c[i, j] = np.exp((p0*center_x+q0*center_y)/2/eps**2)
            p_coeff[i, j] = p0/2/eps**2
            q_coeff[i, j] = q0/2/eps**2

    # Number of unknowns per square (4 coefficients a_i, b_i, c_i, d_i)
    num_unknowns = 4 * N * N
    A = np.zeros((num_unknowns, num_unknowns))
    B = np.zeros(num_unknowns)

    def index(i, j, k):
        return 4 * (i * N + j) + k

    eq_counter = 0

    for i in range(N):
        for j in range(N):
            idx_a = index(i, j, 0)
            idx_b = index(i, j, 1)
            idx_c = index(i, j, 2)
            idx_d = index(i, j, 3)
            mid_x = (j + 0.5) * h
            mid_y = (i + 0.5) * h

            # Boundary conditions (u = 0)
            if i == 0:  # Bottom boundary
                mux, muy = mu[i, j] * (mid_x-mid_x), mu[i, j] * (0-mid_y)
                c = np.exp(p_coeff[i, j]*(mid_x-mid_x) + q_coeff[i, j]*(0-mid_y))
                A[eq_counter, idx_a] = np.exp(mux) * c
                A[eq_counter, idx_b] = np.exp(-mux) * c
                A[eq_counter, idx_c] = np.exp(muy) * c
                A[eq_counter, idx_d] = np.exp(-muy) * c
                B[eq_counter] = -nu[i, j]
                eq_counter += 1
            if i == N-1:  # Top boundary
                mux, muy = mu[i, j] * (mid_x-mid_x), mu[i, j] * (1-mid_y)
                c = np.exp(p_coeff[i, j]*(mid_x-mid_x) + q_coeff[i, j]*(1-mid_y))
                A[eq_counter, idx_a] = np.exp(mux) * c
                A[eq_counter, idx_b] = np.exp(-mux) * c
                A[eq_counter, idx_c] = np.exp(muy) * c
                A[eq_counter, idx_d] = np.exp(-muy) * c
                B[eq_counter] = -nu[i, j]
                eq_counter += 1
            if j == 0:  # Left boundary
                mux, muy = mu[i, j] * (0-mid_x), mu[i, j] * (mid_y-mid_y)
                c = np.exp(p_coeff[i, j]*(0-mid_x) + q_coeff[i, j]*(mid_y-mid_y))
                A[eq_counter, idx_a] = np.exp(mux) * c
                A[eq_counter, idx_b] = np.exp(-mux) * c
                A[eq_counter, idx_c] = np.exp(muy) * c
                A[eq_counter, idx_d] = np.exp(-muy) * c
                B[eq_counter] = -nu[i, j]
                eq_counter += 1
            if j == N-1:  # Right boundary
                mux, muy = mu[i, j] * (1-mid_x), mu[i, j] * (mid_y-mid_y)
                c = np.exp(p_coeff[i, j]*(1-mid_x) + q_coeff[i, j]*(mid_y-mid_y))
                A[eq_counter, idx_a] = np.exp(mux) * c
                A[eq_counter, idx_b] = np.exp(-mux) * c
                A[eq_counter, idx_c] = np.exp(muy) * c
                A[eq_counter, idx_d] = np.exp(-muy) * c
                B[eq_counter] = -nu[i, j]
                eq_counter += 1

            if i < N-1:  # Continuity in the y-direction with the top neighbor
                idx_a_top = index(i+1, j, 0)
                idx_b_top = index(i+1, j, 1)
                idx_c_top = index(i+1, j, 2)
                idx_d_top = index(i+1, j, 3)

                x, y = mid_x, mid_y+h/2
                mux, muy = mu[i, j] * (x-mid_x), mu[i, j] * (y-mid_y)
                c = np.exp(p_coeff[i, j]*(x-mid_x) + q_coeff[i, j]*(y-mid_y))
                A[eq_counter, idx_a] = c * np.exp(mux)
                A[eq_counter, idx_b] = c * np.exp(-mux)
                A[eq_counter, idx_c] = c * np.exp(muy)
                A[eq_counter, idx_d] = c * np.exp(-muy)
                A[eq_counter+1, idx_a] = c * q_coeff[i, j] * np.exp(mux)
                A[eq_counter+1, idx_b] = c * q_coeff[i, j] * np.exp(-mux)
                A[eq_counter+1, idx_c] = c * (q_coeff[i, j] + mu[i, j])*  np.exp(muy)
                A[eq_counter+1, idx_d] = c * (q_coeff[i, j] - mu[i, j]) * np.exp(-muy)

                mux, muy = mu[i+1, j] * (x-mid_x), mu[i+1, j] * (y-mid_y)
                c = np.exp(p_coeff[i+1, j]*(x-mid_x) + q_coeff[i+1, j]*(y-mid_y))
                A[eq_counter, idx_a_top] = -c * np.exp(mux)
                A[eq_counter, idx_b_top] = -c * np.exp(-mux)
                A[eq_counter, idx_c_top] = -c * np.exp(muy)
                A[eq_counter, idx_d_top] = -c * np.exp(-muy)
                A[eq_counter+1, idx_a_top] = -c * q_coeff[i+1, j] * np.exp(mux)
                A[eq_counter+1, idx_b_top] = -c * q_coeff[i+1, j] * np.exp(-mux)
                A[eq_counter+1, idx_c_top] = -c * (q_coeff[i+1, j] + mu[i+1, j]) * np.exp(muy)
                A[eq_counter+1, idx_d_top] = -c * (q_coeff[i+1, j] - mu[i+1, j]) * np.exp(-muy)
                B[eq_counter] = nu[i+1, j] - nu[i, j]
                B[eq_counter+1] = nu[i+1, j] - nu[i, j]
                eq_counter += 2


            if j < N-1:  # Continuity in the x-direction with the right neighbor
                idx_a_right = index(i, j+1, 0)
                idx_b_right = index(i, j+1, 1)
                idx_c_right = index(i, j+1, 2)
                idx_d_right = index(i, j+1, 3)

                x, y = mid_x+h/2, mid_y
                mux, muy = mu[i, j] * (x-mid_x), mu[i, j] * (y-mid_y)
                c = np.exp(p_coeff[i, j]*(x-mid_x) + q_coeff[i, j]*(y-mid_y))
                A[eq_counter, idx_a] = c * np.exp(mux)
                A[eq_counter, idx_b] = c * np.exp(-mux)
                A[eq_counter, idx_c] = c * np.exp(muy)
                A[eq_counter, idx_d] = c * np.exp(-muy)
                A[eq_counter+1, idx_a] = c * q_coeff[i, j] * np.exp(mux)
                A[eq_counter+1, idx_b] = c * q_coeff[i, j] * np.exp(-mux)
                A[eq_counter+1, idx_c] = c * (q_coeff[i, j] + mu[i, j])*  np.exp(muy)
                A[eq_counter+1, idx_d] = c * (q_coeff[i, j] - mu[i, j]) * np.exp(-muy)

                mux, muy = mu[i, j+1] * (x-mid_x), mu[i, j+1] * (y-mid_y)
                c = np.exp(p_coeff[i, j+1]*(x-mid_x) + q_coeff[i, j+1]*(y-mid_y))
                A[eq_counter, idx_a_right] = -c * np.exp(mux)
                A[eq_counter, idx_b_right] = -c * np.exp(-mux)
                A[eq_counter, idx_c_right] = -c * np.exp(muy)
                A[eq_counter, idx_d_right] = -c * np.exp(-muy)
                A[eq_counter+1, idx_a_right] = -c * q_coeff[i, j+1] * np.exp(mux)
                A[eq_counter+1, idx_b_right] = -c * q_coeff[i, j+1] * np.exp(-mux)
                A[eq_counter+1, idx_c_right] = -c * (q_coeff[i, j+1] + mu[i, j+1]) * np.exp(muy)
                A[eq_counter+1, idx_d_right] = -c * (q_coeff[i, j+1] - mu[i, j+1]) * np.exp(-muy)
                B[eq_counter] = nu[i, j+1] - nu[i, j]
                B[eq_counter+1] = nu[i, j+1] - nu[i, j]
                eq_counter += 2

    return A, B


N = 1
eps = 0.1
p = lambda x, y: 1.0
q = lambda x, y: 1.0
b = lambda x, y: 1.0
f = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)

A, B = create_linear_system(N, eps, p, q, b, f)
M_inverse = np.diag(1 / np.max(np.abs(A), axis=0))
A = np.matmul(A, M_inverse)
x = solve(A, B)


print(x.reshape(N, N, 4))
