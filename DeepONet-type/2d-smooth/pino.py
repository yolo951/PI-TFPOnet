
import sys
sys.path.append('D:/pythonProject/TFP-Net/DeepONet-type')
import matplotlib.pyplot as plt
from FNO2d import FNO2d
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import numpy as np
from Adam import Adam
from timeit import default_timer

@torch.jit.script
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    res =  torch.einsum("bxy,xy->bxy", a, b)
    return res

# Based on https://github.com/neuraloperator/physics_informed, we got the following code
# In order to more conveniently obtain the conditions at the interface, we use two networks to predict the solutions in the two regions respectively.
def FDM(u_left, u_right, eps=1., L=1):
    N = u_left.size(1) - 1
    dx = L / N
    dy = dx

    # ux: (batch, size-2, size-2)
    # The following is the method to obtain the first-order gradient on the interface
    # In the area to the left of the interface, we use the reverse Euler. 
    # In the area to the right of the interface, we use the forward Euler.
    uxx_left = (u_left[:, 1:-1, 2:] - 2*u_left[:, 1:-1, 1:-1] + u_left[:, 1:-1, :-2]) / (dx * dx)
    uyy_left = (u_left[:, 2:, 1:-1] - 2*u_left[:, 1:-1, 1:-1] + u_left[:, :-2, 1:-1]) / (dy * dy)
    Du_left = - eps*(uxx_left + uyy_left)
    ux_left = (u_left[:, :, -1] - u_left[:, :, -2]) / dx

    uxx_right = (u_right[:, 1:-1, 2:] - 2*u_right[:, 1:-1, 1:-1] + u_right[:, 1:-1, :-2]) / (dx * dx)
    uyy_right = (u_right[:, 2:, 1:-1] - 2*u_right[:, 1:-1, 1:-1] + u_right[:, :-2, 1:-1]) / (dy * dy)
    Du_right = - eps*(uxx_right + uyy_right)
    ux_right = (u_right[:, :, 1] - u_right[:, :, 0]) / dx
    grads = {'Du_left': Du_left, 'Du_right': Du_right, 'ux_left': ux_left, 'ux_right': ux_right}
    return grads

def PINO_loss(u_left, fx_left, u_right, fx_right, myloss, coeffs):
    batch_size = u_left.shape[0]
    u_left = u_left.squeeze(-1)
    u_right = u_right.squeeze(-1)
    N = u_left.size(1)-1
    x = torch.linspace(0, 1, N+1).to(device)
    xx, yy = torch.meshgrid(x, x, indexing="xy")
    c = torch.where(xx<1/2, 16., 1.)
    grads = FDM(u_left, u_right, eps=1.)
    idx_left = torch.LongTensor([[0, i] for i in range(N//2+1)]
                                             +[[i, 0] for i in range(1, N)]
                                             +[[N, i] for i in range(1, N//2+1)]).to(device)
    idx_right = torch.LongTensor([[0, i] for i in range(N//2+1)]
                                 +[[i, -1] for i in range(1, N)]
                                 +[[N, i] for i in range(N//2)]).to(device)
    ub_left = 1-yy[:, :N//2+1].unsqueeze(0).repeat(batch_size, 1, 1)
    ub_right = -yy[:, N//2:].unsqueeze(0).repeat(batch_size, 1, 1)
    Du_left, Du_right, ux_left, ux_right = grads['Du_left'], grads['Du_right'], grads['ux_left'], grads['ux_right']
    coeff_equ, coeff_i, coeff_i_grad, coeff_b = coeffs['coeff_equ'], coeffs['coeff_i'], coeffs['coeff_i_grad'], coeffs['coeff_b']

    # equation: -eps*(u_xx+u_yy) + c*u = f
    cu_left = matmul(u_left[:, 1:-1, 1:-1].squeeze(-1), c[1:-1, 1:N//2])
    cu_right = matmul(u_right[:, 1:-1, 1:-1].squeeze(-1), c[1:-1, N//2+1:-1])
    loss_equ = myloss(Du_left+cu_left, fx_left[:, 1:-1, 1:-1, 0]) + myloss(Du_right+cu_right, fx_right[:, 1:-1, 1:-1, 0])
    loss_i = myloss(u_right[:, :, 0]-1, u_left[:, :, -1])
    loss_i_grad = myloss(ux_right, ux_left)
    loss_b = myloss(u_left[:, idx_left[:, 0], idx_left[:, 1]], ub_left[:, idx_left[:, 0], idx_left[:, 1]])\
          + myloss(u_right[:, idx_right[:, 0], idx_right[:, 1]], ub_right[:, idx_right[:, 0], idx_right[:, 1]])
    loss = coeff_equ*loss_equ + coeff_i*loss_i + coeff_i_grad*loss_i_grad + coeff_b*loss_b
    return loss

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    modes1 = [20, 20, 20, 20]
    modes2 = [9, 9, 9, 9]
    width = 64
    epochs = 10000
    learning_rate = 0.001
    step_size = 2000
    gamma = 0.5
    batch_size = 64
    coeffs = {'coeff_equ': 10.0, 'coeff_i': 10.0, 'coeff_i_grad': 10.0, 'coeff_b': 10.0}
    data = np.load("DeepONet-type/2d-smooth/saved_data/data.npz")
    data_fno = np.load("DeepONet-type/2d-smooth/saved_data/data_fno.npz")

    ntrain, ntest = 1000, 200
    N = 32
    M = 4
    alpha = 1.
    beta = 0.

    f_train_sparse = torch.tensor(data['f_total'][:ntrain].reshape((ntrain, N+1, N+1, 1)), dtype=torch.float32)
    x_sparse = torch.linspace(0, 1, N+1, dtype=torch.float32)
    xx, yy = torch.meshgrid((x_sparse, x_sparse), indexing="xy")
    points_sparse = torch.stack((xx, yy), axis=-1).unsqueeze(0).repeat(ntrain, 1, 1, 1)
    input_train = torch.concat((f_train_sparse, points_sparse), dim=-1).to(device)
    u_train_sparse = torch.tensor(data_fno['u_train_sparse'], dtype=torch.float32).to(device)
    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_train, u_train_sparse), batch_size=batch_size, shuffle=True)

    f_test_sparse = torch.tensor(data['f_total'][-ntest:].reshape((ntest, N+1, N+1, 1)), dtype=torch.float32)
    points_sparse = torch.stack((xx, yy), axis=-1).unsqueeze(0).repeat(ntest, 1, 1, 1)
    input_test_sparse = torch.concat((f_test_sparse, points_sparse), dim=-1).to(device)
    u_test_sparse = torch.tensor(data['u_test_fine'][:, ::M, ::M], dtype=torch.float32).to(device)

    f_test_fine = torch.tensor(data_fno['f_test_fine'], dtype=torch.float32).unsqueeze(-1)  # test f on fine grid
    x_fine = torch.linspace(0, 1, N*M+1, dtype=torch.float32)
    xx, yy = torch.meshgrid((x_fine, x_fine), indexing="xy")
    points_fine = torch.stack((xx, yy), axis=-1).unsqueeze(0).repeat(ntest, 1, 1, 1)
    input_test_fine = torch.concat((f_test_fine, points_fine), dim=-1).to(device)
    u_test_fine = torch.tensor(data['u_test_fine'], dtype=torch.float32).to(device)  # test u on fine grid

    model_left = FNO2d(modes1, modes2, width).to(device)
    model_right = FNO2d(modes1, modes2, width).to(device)
    optimizer_left = Adam(model_left.parameters(), betas=(0.9, 0.999), lr=learning_rate)
    optimizer_right = Adam(model_right.parameters(), betas=(0.9, 0.999), lr=learning_rate)
    myloss = torch.nn.MSELoss(reduction='mean')
    scheduler_left = torch.optim.lr_scheduler.MultiStepLR(optimizer_left, milestones=[2000, 5000, 7000, 10000], gamma=gamma)
    scheduler_right = torch.optim.lr_scheduler.MultiStepLR(optimizer_right, milestones=[2000, 5000, 7000, 10000], gamma=gamma)
    mse_history = []
    rel_l2_history = []
    model_left.train()
    model_right.train()
    for ep in range(epochs):
        train_mse = 0.
        t1 = default_timer()
        for fx, u in data_loader:
            optimizer_left.zero_grad()
            optimizer_right.zero_grad()
            fx_left = fx[:, :, :N//2+1]
            fx_right = fx[:, :, N//2:]
            out_left = model_left(fx_left)
            out_right = model_right(fx_right)
            loss = PINO_loss(out_left, fx_left, out_right, fx_right, myloss, coeffs)
            
            loss.backward()
            optimizer_left.step()
            optimizer_right.step()
            train_mse += loss.item()
        scheduler_left.step()
        scheduler_right.step()
        train_mse /= len(data_loader)
        mse_history.append(train_mse)
        if ep==0 or (ep+1)%100==0:
            t2 = default_timer()
            fx_left = input_test_sparse[:, :, :N//2+1]
            fx_right = input_test_sparse[:, :, N//2:]
            u_pred_left = model_left(fx_left)
            u_pred_right = model_right(fx_right)
            up_pred = torch.concat((u_pred_left[:, :, :-1], u_pred_right), dim=-2).squeeze(-1)
            rel_l2 = torch.linalg.norm(up_pred.flatten() - u_test_sparse.flatten()).item() / torch.linalg.norm(u_test_sparse.flatten()).item()
            rel_l2_history.append(rel_l2)
            print('\repoch {:d}/{:d} L2 = {:.6f}, MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, rel_l2, train_mse, t2 - t1), end='\n', flush=True)
    model_left.eval()
    model_right.eval()
    np.save('DeepONet-type/2d-smooth/saved_data/unsupervised_fno_loss_history.npy', mse_history)
    np.save('DeepONet-type/2d-smooth/saved_data/unsupervised_fno_rel_l2_history.npy', rel_l2_history)
    torch.save({'model_left': model_left.state_dict(), 'model_right': model_right.state_dict()}, 'DeepONet-type/2d-smooth/saved_data/unsupervised_fno_model_state.pth')

    with torch.no_grad(): 
        fx_left = input_test_fine[:, :, :N*M//2+1]
        fx_right = input_test_fine[:, :, N*M//2:]
        u_pred_left = model_left(fx_left)
        u_pred_right = model_right(fx_right)
        up_pred = torch.concat((u_pred_left[:, :, :-1], u_pred_right), dim=-2).squeeze(-1)
        print('test error on high resolution: relative L2 norm = ', torch.linalg.norm(up_pred.flatten() - u_test_fine.flatten()).item() / torch.linalg.norm(u_test_fine.flatten()).item())
        print('test error on high resolution: relative L_infty norm = ', torch.linalg.norm(up_pred.flatten() -  u_test_fine.flatten(), ord=torch.inf).item() / torch.linalg.norm(u_test_fine.flatten(), ord=torch.inf).item())
    plt.figure()
    plt.plot(np.arange(0, epochs+1, 100), rel_l2_history, '-*')
    plt.xlabel('epochs')
    plt.ylabel('relative l2 error')
    # plt.ylim(1e-3, 1e+2)
    plt.yscale("log")
    plt.savefig('DeepONet-type/2d-smooth/saved_data/unsupervised_fno_l2.png')
    plt.show()