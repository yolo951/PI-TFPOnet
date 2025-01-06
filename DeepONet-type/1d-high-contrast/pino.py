
import sys
sys.path.append('DeepONet-type')
import matplotlib.pyplot as plt
from FNO1d import FNO1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import numpy as np
from Adam import Adam
from timeit import default_timer

# Based on https://github.com/neuraloperator/physics_informed, we got the following code
# In order to more conveniently obtain the conditions at the interface, we use two networks to predict the solutions in the two regions respectively.
class PINO_loss:
    def __init__(self, eps_left, eps_right, batch_size, N, coeffs, myloss, device):
        x_left = torch.linspace(0, 0.5, N//2+1).to(device)
        x_right = torch.linspace(0.5, 1, N//2+1).to(device)
        self.q_left = 2*x_left+1
        self.q_right = 2*(1-x_right)+1
        self.ub = torch.zeros((batch_size, 1), dtype=torch.float32).squeeze(-1).to(device)
        self.eps_left = eps_left
        self.eps_right = eps_right
        self.coeffs = coeffs
        self.myloss = myloss

    @staticmethod
    def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        res =  torch.einsum("bx,x->bx", a, b)
        return res

    def FDM(self, u_left, u_right, L=1):
        N = u_left.size(1) - 1
        dx = L / N

        # The following is the method to obtain the first-order gradient on the interface
        # In the area to the left of the interface, we use the reverse Euler. 
        # In the area to the right of the interface, we use the forward Euler.
        uxx_left = (u_left[:, 2:] - 2*u_left[:, 1:-1] + u_left[:, :-2]) / (dx * dx)
        Du_left = - self.eps_left*uxx_left
        ux_left = (u_left[:, -1] - u_left[:, -2]) / dx

        uxx_right = (u_right[:, 2:] - 2*u_right[:, 1:-1] + u_right[:, :-2]) / (dx * dx)
        Du_right = - self.eps_right*uxx_right
        ux_right = (u_right[:, 1] - u_right[:, 0]) / dx
        grads = {'Du_left': Du_left, 'Du_right': Du_right, 'ux_left': ux_left, 'ux_right': ux_right}
        return grads

    def compute_loss(self, u_left, fx_left, u_right, fx_right):
        u_left = u_left.squeeze(-1)
        u_right = u_right.squeeze(-1)
        grads = self.FDM(u_left, u_right)
        Du_left, Du_right, ux_left, ux_right = grads['Du_left'], grads['Du_right'], grads['ux_left'], grads['ux_right']
        coeff_equ, coeff_i, coeff_i_grad, coeff_b = coeffs['coeff_equ'], coeffs['coeff_i'], coeffs['coeff_i_grad'], coeffs['coeff_b']

        # equation: -eps*u_'' + q*u = f
        cu_left = self.matmul(u_left[:, 1:-1].squeeze(-1), self.q_left[1:-1])
        cu_right = self.matmul(u_right[:, 1:-1].squeeze(-1), self.q_right[1:-1])
        loss_equ = myloss(Du_left+cu_left, fx_left[:, 1:-1, 0]) + myloss(Du_right+cu_right, fx_right[:, 1:-1, 0])
        loss_i = myloss(u_right[:, 0]-1, u_left[:, -1])
        loss_i_grad = myloss(ux_right, ux_left)
        loss_b = myloss(u_left[:, 0], self.ub) + myloss(u_right[:, -1], self.ub)
        loss = coeff_equ*loss_equ + coeff_i*loss_i + coeff_i_grad*loss_i_grad + coeff_b*loss_b
        return loss

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    modes = [9, 9, 9, 9]
    width = 64
    epochs = 3000
    learning_rate = 0.001
    gamma = 0.6
    step_size = 600
    batch_size = 100
    coeffs = {'coeff_equ': 1.0, 'coeff_i': 10.0, 'coeff_i_grad': 10.0, 'coeff_b': 1.0}
    f = np.load('DeepONet-type/1d-high-contrast/f.npy')

    ntrain, ntest = 1000, 200
    N = 32
    alpha = 1.
    beta = 0.

    f_train_sparse = torch.tensor(f[:ntrain].reshape((ntrain, N+1, 1)), dtype=torch.float32)
    x_sparse = torch.linspace(0, 1, N+1, dtype=torch.float32).reshape((-1, 1))
    points_sparse = x_sparse.unsqueeze(0).repeat(ntrain, 1, 1)
    input_train = torch.concat((f_train_sparse, points_sparse), dim=-1).to(device)
    data_loader = torch.utils.data.DataLoader(input_train, batch_size=batch_size, shuffle=True)

    model_left = FNO1d(modes, width).to(device)
    model_right = FNO1d(modes, width).to(device)
    optimizer_left = Adam(model_left.parameters(), betas=(0.9, 0.999), lr=learning_rate)
    optimizer_right = Adam(model_right.parameters(), betas=(0.9, 0.999), lr=learning_rate)
    myloss = torch.nn.MSELoss(reduction='mean')
    scheduler_left = torch.optim.lr_scheduler.StepLR(optimizer_left, step_size=step_size, gamma=gamma)
    scheduler_right = torch.optim.lr_scheduler.StepLR(optimizer_right, step_size=step_size, gamma=gamma)
    pino_loss = PINO_loss(0.001, 1., batch_size, N, coeffs, myloss, device)
    model_left.train()
    model_right.train()
    for ep in range(epochs):
        train_mse = 0.
        t1 = default_timer()
        for fx in data_loader:
            optimizer_left.zero_grad()
            optimizer_right.zero_grad()
            fx_left = fx[:, :N//2+1]
            fx_right = fx[:, N//2:]
            out_left = model_left(fx_left)
            out_right = model_right(fx_right)
            loss = pino_loss.compute_loss(out_left, fx_left, out_right, fx_right)
            
            loss.backward()
            optimizer_left.step()
            optimizer_right.step()
            train_mse += loss.item()
        scheduler_left.step()
        scheduler_right.step()
        train_mse /= len(data_loader)
        t2 = default_timer()
        print('\repoch {:d}/{:d}, MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse, t2 - t1), end='\n', flush=True)

    model_left.eval()
    model_right.eval()

    N_fine = 256
    f_test_fine = np.load('DeepONet-type/1d-high-contrast/f_test_fine.npy')
    u_test_fine = np.load('DeepONet-type/1d-high-contrast/u_test.npy')
    f_test_fine = torch.tensor(f_test_fine, dtype=torch.float32).unsqueeze(-1)  # test f on fine grid
    x_fine = torch.linspace(0, 1, N_fine+1, dtype=torch.float32)
    points_fine = x_fine.reshape((-1, 1)).unsqueeze(0).repeat(ntest, 1, 1)
    input_test_fine = torch.concat((f_test_fine, points_fine), dim=-1).to(device)
    u_test_fine = torch.tensor(u_test_fine, dtype=torch.float32).to(device)  # test u on fine grid
    with torch.no_grad(): 
        fx_left = input_test_fine[:, :N_fine//2+1]
        fx_right = input_test_fine[:, N_fine//2:]
        u_pred_left = model_left(fx_left)
        u_pred_right = model_right(fx_right)
        up_pred = torch.concat((u_pred_left[:, :-1], u_pred_right), dim=-2).squeeze(-1)
        print('test error on high resolution: relative L2 norm = ', torch.linalg.norm(up_pred.flatten() - u_test_fine.flatten()).item() / torch.linalg.norm(u_test_fine.flatten()).item())
        print('test error on high resolution: relative L_infty norm = ', torch.linalg.norm(up_pred.flatten() -  u_test_fine.flatten(), ord=torch.inf).item() / torch.linalg.norm(u_test_fine.flatten(), ord=torch.inf).item())

    u_test_fine = u_test_fine.cpu().numpy()
    up_pred = up_pred.detach().cpu().numpy()
    fig = plt.figure(figsize=(4, 3), dpi=150)
    plt.plot(x_fine[:int(N_fine/2+1)], u_test_fine[-ntest, :int(N_fine/2+1)].flatten(), 'b-', label='Ground Truth', linewidth=2, alpha=1., zorder=0)
    plt.plot(x_fine[int(N_fine/2+1):], u_test_fine[-ntest, int(N_fine/2+1):].flatten(), 'b-', linewidth=2, alpha=1., zorder=0)
    plt.plot(x_fine[:int(N_fine/2+1)], up_pred[-ntest, :int(N_fine/2+1)], 'r--', label='Prediction', linewidth=2, alpha=1., zorder=0)
    plt.plot(x_fine[int(N_fine/2+1):], up_pred[-ntest, int(N_fine/2+1):], 'r--', linewidth=2, alpha=1., zorder=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('DeepONet-type/1d-high-contrast/1d_fno_example.png')