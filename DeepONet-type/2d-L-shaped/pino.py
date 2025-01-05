
import sys
sys.path.append('DeepONet-type')
import matplotlib.pyplot as plt
from GeoFNO2d import FNO2d, IPHI
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import numpy as np
from Adam import Adam
from timeit import default_timer
torch.autograd.set_detect_anomaly(True)

@torch.jit.script
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res =  torch.einsum("bx,x->bx", a, b)
    return res

def generate_index(N):
    # idx except interface
    idx_y = torch.cat((
        torch.tensor([[N // 4 + i] * (N // 2 - 1) for i in range(1, N // 4)]).flatten(),
        torch.tensor([[N // 2 + i] * (N // 4 - 1) for i in range(0, N // 4)]).flatten()))
    idx_y_remain = torch.cat((
        torch.tensor([[i] * (N - 1) for i in range(1, N // 4)]).flatten(),
        torch.tensor([[N // 4 + i] * (N // 2 - 2) for i in range(0, N // 4 + 1)]).flatten(),
        torch.tensor([[N // 2 + i] * (N * 3 // 4 - 2) for i in range(1, N // 4 + 1)]).flatten(),
        torch.tensor([[N * 3 // 4 + i] * (N - 1) for i in range(1, N // 4)]).flatten()))

    idx_x = torch.cat((
        torch.tensor([N // 4 + j for j in range(1, N // 2)]).repeat(N // 4 - 1).flatten(),
        torch.tensor([N // 4 + j for j in range(1, N // 4)]).repeat(N // 4).flatten()))
    idx_x_remain = torch.cat((
        torch.arange(1, N).repeat(N // 4 - 1).flatten(),
        torch.tensor([j for j in range(1, N // 4)] + [j for j in range(N * 3 // 4 + 1, N)]).repeat(N // 4 + 1).flatten(),
        torch.tensor([j for j in range(1, N // 4)] + [j for j in range(N // 2 + 1, N)]).repeat(N // 4).flatten(),
        torch.arange(1, N).repeat(N // 4 - 1).flatten()))

    idx_mask_inner = torch.stack((idx_x, idx_y), dim=-1).long()
    idx_mask_outer = torch.stack((idx_x_remain, idx_y_remain), dim=-1).long()
    mask = {'idx_inner': idx_mask_inner, 'idx_outer': idx_mask_outer}

    # for u with shape=(batch_size, N+1, N+1)
    idx_inner = torch.LongTensor([[i, N//4+1] for i in range(N//4+1, N*3//4)] 
                                 + [[N*3//4-1, j] for j in range(N//4+1, N//2)] 
                                 + [[i, N//2-1] for i in range(N//2+1, N*3//4)] 
                                 + [[N//2-1, j] for j in range(N//2+1, N*3//4)] 
                                 + [[i, N*3//4-1] for i in range(N//4+1, N//2)] 
                                 + [[N//4+1, j] for j in range(N//4+1, N*3//4)])
    idx_interface = torch.LongTensor([[i, N//4] for i in range(N//4+1, N*3//4)] 
                                     + [[N*3//4, j] for j in range(N//4+1, N//2)] 
                                     + [[i, N//2] for i in range(N//2+1, N*3//4)] 
                                     + [[N//2, j] for j in range(N//2+1, N*3//4)] 
                                     + [[i, N*3//4] for i in range(N//4+1, N//2)] 
                                     + [[N//4, j] for j in range(N//4+1, N*3//4)])
    idx_outer = torch.LongTensor([[i, N//4-1] for i in range(N//4+1, N*3//4)] 
                                 + [[N*3//4+1, j] for j in range(N//4+1, N//2)] 
                                 + [[i, N//2+1] for i in range(N//2+1, N*3//4)] 
                                 + [[N//2+1, j] for j in range(N//2+1, N*3//4)] 
                                 + [[i, N*3//4+1] for i in range(N//4+1, N//2)] 
                                 + [[N//4-1, j] for j in range(N//4+1, N*3//4)])
    idx_boundary = torch.LongTensor([[i, 0] for i in range(N+1)] 
                                    + [[N, j] for j in range(N+1)] 
                                    + [[i, N] for i in range(N+1)] 
                                    + [[0, j] for j in range(N+1)])
    line = {'idx_inner': idx_inner, 'idx_interface': idx_interface, 'idx_outer': idx_outer, 'idx_boundary': idx_boundary}
    
    idx_corner = torch.LongTensor([[N//4, N//4], [N*3//4, N//4], [N*3//4, N//2], [N//2, N//2], [N//2, N*3//4], [N//4, N*3//4]])
    idx_closure_inner = torch.cat((idx_interface, idx_mask_inner, idx_corner), dim=0)
    idx_closure_outer = torch.cat((idx_boundary, idx_mask_outer, idx_interface, idx_corner), dim=0)
    closure = {'idx_inner': idx_closure_inner, 'idx_outer': idx_closure_outer}
    return mask, line, closure

# Based on https://github.com/neuraloperator/physics_informed, we got the following code
# In order to more conveniently obtain the conditions at the interface, we use two networks to predict the solutions in the two regions respectively.
def FDM(u, line, eps=1., L=1):
    N = u.size(1) - 1
    dx = L / N
    dy = dx
    uxx = (u[:, 1:-1, 2:] - 2*u[:, 1:-1, 1:-1] + u[:, 1:-1, :-2]) / (dx * dx)
    uyy = (u[:, 2:, 1:-1] - 2*u[:, 1:-1, 1:-1] + u[:, :-2, 1:-1]) / (dy * dy)
    Du = - eps*(uxx + uyy)

    idx_inner, idx_interface, idx_outer = line['idx_inner'], line['idx_interface'], line['idx_outer']
    du_inner = (u[:, idx_interface[:, 1], idx_interface[:, 0]]-u[:, idx_inner[:, 1], idx_inner[:, 0]])/dx
    du_outer = (u[:, idx_outer[:, 1], idx_outer[:, 0]]-u[:, idx_interface[:, 1], idx_interface[:, 0]])/dx
    grads = {'Du': Du, 'du_inner': du_inner, 'du_outer': du_outer}
    return grads
    
def PINO_loss(u_inner, u_outer, f, mask, line, closure, myloss, coeffs):
    idx_mask_inner, idx_mask_outer = mask['idx_inner'], mask['idx_outer']
    idx_closure_inner, idx_closure_outer = closure['idx_inner'], closure['idx_outer']
    idx_interface, idx_boundary = line['idx_interface'], line['idx_boundary']
    batch_size = f.shape[0]
    N = f.shape[1]-1
    u_full_inner = torch.zeros((batch_size, N+1, N+1), dtype=torch.float32).to(f.device)
    u_full_outer = torch.zeros((batch_size, N+1, N+1), dtype=torch.float32).to(f.device)
    u_full_inner[:, idx_closure_inner[:, 1], idx_closure_inner[:, 0]] = u_inner.squeeze(-1)
    u_full_outer[:, idx_closure_outer[:, 1], idx_closure_outer[:, 0]] = u_outer.squeeze(-1)

    x = torch.linspace(0, 1, N+1).to(device)
    xx, yy = torch.meshgrid(x, x, indexing="xy")
    c = torch.where(((0.25 <= yy) & (yy <= 0.5) & (0.25 <= xx) & (xx <= 0.75)) |
                                ((0.5 <= yy) & (yy <= 0.75) & (0.25 <= xx) & (xx <= 0.5)), 1.0, 16.0)
    ub = 0.5*torch.where(yy == 1, 1 - xx, torch.where(yy == 0, xx, torch.where(xx == 1, 1 - yy, yy))).unsqueeze(0).repeat(batch_size, 1, 1)

    grads = FDM(u_full_inner, line, eps=0.001)
    Du, du_inner = grads['Du'], grads['du_inner']
    temp = torch.zeros_like(f, dtype=f.dtype, device=f.device)
    temp[:, 1:-1, 1:-1] = Du.unsqueeze(-1)
    Du_inner = temp[:, idx_mask_inner[:, 1], idx_mask_inner[:, 0]]
    grads = FDM(u_full_outer, line, eps=0.001)
    Du, du_outer = grads['Du'], grads['du_outer']
    temp[:, 1:-1, 1:-1] = Du.unsqueeze(-1)
    Du_outer = temp[:, idx_mask_outer[:, 1], idx_mask_outer[:, 0]]
    cu_inner = matmul(u_full_inner[:, idx_mask_inner[:, 1], idx_mask_inner[:, 0]], c[idx_mask_inner[:, 1], idx_mask_inner[:, 0]]).unsqueeze(-1)
    cu_outer = matmul(u_full_outer[:, idx_mask_outer[:, 1], idx_mask_outer[:, 0]], c[idx_mask_outer[:, 1], idx_mask_outer[:, 0]]).unsqueeze(-1)
    loss_equ = myloss(Du_inner+cu_inner, f[:, idx_mask_inner[:, 1], idx_mask_inner[:, 0]])+myloss(Du_outer+cu_outer, f[:, idx_mask_outer[:, 1], idx_mask_outer[:, 0]])
    loss_i = myloss(u_full_inner[:, idx_interface[:, 1], idx_interface[:, 0]], u_full_outer[:, idx_interface[:, 1], idx_interface[:, 0]]+1)
    loss_i_grad = myloss(du_inner, du_outer)
    loss_b = myloss(u_full_outer[:, idx_boundary[:, 1], idx_boundary[:, 0]], ub[:, idx_boundary[:, 1], idx_boundary[:, 0]])
    coeff_equ, coeff_i, coeff_i_grad, coeff_b = coeffs['coeff_equ'], coeffs['coeff_i'], coeffs['coeff_i_grad'], coeffs['coeff_b']
    loss = coeff_equ*loss_equ + coeff_i*loss_i + coeff_i_grad*loss_i_grad + coeff_b*loss_b
    return loss

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    modes1 = 16
    modes2 = 16
    width = 32
    epochs = 2000
    learning_rate = 0.001
    gamma = 0.5
    batch_size = 200
    coeffs = {'coeff_equ': 0.05, 'coeff_i': 1.0, 'coeff_i_grad': 10.0, 'coeff_b': 50.0}
    data = np.load("DeepONet-type/2d-L-shaped/saved_data/data.npz")
    data_fno = np.load("DeepONet-type/2d-L-shaped/saved_data/data_fno.npz")

    ntrain, ntest = 1000, 200
    N = 32
    M = 4
    alpha = 1.
    beta = 0.

    mask, line, closure = generate_index(N)
    idx_closure_inner, idx_closure_outer = closure['idx_inner'], closure['idx_outer']
    f_train_sparse = torch.tensor(data['f_total'][:ntrain].reshape((ntrain, N+1, N+1, 1)), dtype=torch.float32)/1000.
    x_sparse = torch.linspace(0, 1, N+1, dtype=torch.float32)
    xx, yy = torch.meshgrid((x_sparse, x_sparse), indexing="xy")
    points_sparse = torch.stack((xx, yy), axis=-1).unsqueeze(0).repeat(ntrain, 1, 1, 1)
    input_train = torch.concat((f_train_sparse, points_sparse), dim=-1).to(device)
    f_train_sparse = f_train_sparse.to(device)
    u_train_sparse = torch.tensor(data_fno['u_train_sparse'], dtype=torch.float32).to(device)

    input_train_inner = input_train[:, idx_closure_inner[:, 1], idx_closure_inner[:, 0], :]
    points_sparse_inner = points_sparse[:, idx_closure_inner[:, 1], idx_closure_inner[:, 0], :]
    u_train_inner = u_train_sparse[:, idx_closure_inner[:, 1], idx_closure_inner[:, 0]]

    input_train_outer = input_train[:, idx_closure_outer[:, 1], idx_closure_outer[:, 0], :]
    points_sparse_outer = points_sparse[:, idx_closure_outer[:, 1], idx_closure_outer[:, 0], :]
    u_train_outer = u_train_sparse[:, idx_closure_outer[:, 1], idx_closure_outer[:, 0]]

    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_train_sparse, input_train_inner, u_train_inner, input_train_outer, input_train_outer), batch_size=batch_size, shuffle=True)

    f_test_sparse = torch.tensor(data['f_total'][-ntest:].reshape((ntest, N+1, N+1, 1)), dtype=torch.float32)/1000.
    points_sparse = torch.stack((xx, yy), axis=-1).unsqueeze(0).repeat(ntest, 1, 1, 1)
    input_test_sparse = torch.concat((f_test_sparse, points_sparse), dim=-1).to(device)
    u_test_sparse = torch.tensor(data['u_test_fine'][:, ::M, ::M], dtype=torch.float32).to(device)
    u_test_sparse = torch.concat((u_test_sparse[:, idx_closure_inner[:, 1], idx_closure_inner[:, 0]],
                                  u_test_sparse[:, idx_closure_outer[:, 1], idx_closure_outer[:, 0]]), dim=-1)

    model_inner = FNO2d(modes1, modes2, width, in_channels=3, out_channels=1).to(device)
    model_inner_iphi = IPHI().to(device)
    optimizer_inner = Adam(model_inner.parameters(), betas=(0.9, 0.999), lr=learning_rate)
    optimizer_inner_iphi = Adam(model_inner_iphi.parameters(), betas=(0.9, 0.999), lr=learning_rate)
    scheduler_inner = torch.optim.lr_scheduler.MultiStepLR(optimizer_inner, milestones=[500, 1000, 3500, 7000, 10000], gamma=gamma)
    scheduler_inner_iphi = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_inner_iphi, T_max = 200)

    model_outer = FNO2d(modes1, modes2, width, in_channels=3, out_channels=1).to(device)
    model_outer_iphi = IPHI().to(device)
    optimizer_outer = Adam(model_outer.parameters(), betas=(0.9, 0.999), lr=learning_rate)
    optimizer_outer_iphi = Adam(model_outer_iphi.parameters(), betas=(0.9, 0.999), lr=learning_rate)
    scheduler_outer = torch.optim.lr_scheduler.MultiStepLR(optimizer_outer, milestones=[500, 1000, 3500, 7000, 10000], gamma=gamma)
    scheduler_outer_iphi = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_outer_iphi, T_max = 200)

    myloss = torch.nn.MSELoss(reduction='mean')
    mse_history = []
    rel_l2_history = []
    model_inner.train()
    model_outer.train()
    for ep in range(epochs):
        train_mse = 0.
        t1 = default_timer()
        for f, fx_i, u_i, fx_o, u_o in data_loader:
            optimizer_inner.zero_grad()
            optimizer_inner_iphi.zero_grad()
            optimizer_outer.zero_grad()
            optimizer_outer_iphi.zero_grad()
            out_i = model_inner(fx_i, x_in=fx_i[:, :, 1:], x_out=fx_i[:, :, 1:], iphi=model_inner_iphi)
            out_o = model_outer(fx_o, x_in=fx_o[:, :, 1:], x_out=fx_o[:, :, 1:], iphi=model_outer_iphi)
            loss = PINO_loss(out_i, out_o, f, mask, line, closure, myloss, coeffs)
            loss.backward()
            optimizer_inner.step()
            optimizer_inner_iphi.step()
            optimizer_outer.step()
            optimizer_outer_iphi.step()
            train_mse += loss.item()
        scheduler_inner.step()
        scheduler_inner_iphi.step()
        scheduler_outer.step()
        scheduler_outer_iphi.step()
        train_mse /= len(data_loader)
        mse_history.append(train_mse)
        if ep==0 or (ep+1)%100==0:
            t2 = default_timer()
            fx = input_test_sparse[:, idx_closure_inner[:, 1], idx_closure_inner[:, 0]]
            x = fx[:, :, 1:]
            u_pred_inner = model_inner(fx, x_in=x, x_out=x, iphi=model_inner_iphi)
            fx = input_test_sparse[:, idx_closure_outer[:, 1], idx_closure_outer[:, 0]]
            x = fx[:, :, 1:]
            u_pred_outer = model_outer(fx, x_in=x, x_out=x, iphi=model_outer_iphi)
            up_pred = torch.concat((u_pred_inner, u_pred_outer), dim=-2).squeeze(-1)
            rel_l2 = torch.linalg.norm(up_pred.flatten() - u_test_sparse.flatten()).item() / torch.linalg.norm(u_test_sparse.flatten()).item()
            rel_l2_history.append(rel_l2)
            print('\repoch {:d}/{:d} L2 = {:.6f}, MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, rel_l2, train_mse, t2 - t1), end='\n', flush=True)
    
    np.save('DeepONet-type/2d-L-shaped/saved_data/unsupervised_fno_loss_history.npy', mse_history)
    np.save('DeepONet-type/2d-L-shaped/saved_data/unsupervised_fno_rel_l2_history.npy', rel_l2_history)
    torch.save({'model_inner': model_inner.state_dict(), 'model_outer': model_outer.state_dict(), 'model_inner_iphi': model_inner_iphi.state_dict(), 'model_outer_iphi': model_outer_iphi.state_dict()}, 'DeepONet-type/2d-L-shaped/saved_data/unsupervised_fno_model_state.pth')

    model_inner = model_inner.to("cpu")
    model_inner_iphi = model_inner_iphi.to("cpu")
    model_outer = model_outer.to("cpu")
    model_outer_iphi = model_outer_iphi.to("cpu")
    _, _, closure = generate_index(N*M)
    idx_closure_inner, idx_closure_outer = closure['idx_inner'], closure['idx_outer']
    f_test_fine = torch.tensor(data_fno['f_test_fine'], dtype=torch.float32).unsqueeze(-1)/1000.  # test f on fine grid
    x_fine = torch.linspace(0, 1, N*M+1, dtype=torch.float32)
    xx, yy = torch.meshgrid((x_fine, x_fine), indexing="xy")
    points_fine = torch.stack((xx, yy), axis=-1).unsqueeze(0).repeat(ntest, 1, 1, 1)
    input_test_fine = torch.concat((f_test_fine, points_fine), dim=-1)
    u_test_fine = torch.tensor(data['u_test_fine'], dtype=torch.float32)  # test u on fine grid
    u_test_fine_inner = u_test_fine[:, idx_closure_inner[:, 1], idx_closure_inner[:, 0]]
    u_test_fine_outer = u_test_fine[:, idx_closure_outer[:, 1], idx_closure_outer[:, 0]]
    u_test_fine = torch.concat((u_test_fine_inner, u_test_fine_outer), dim=-1)

    model_inner.eval()
    model_outer.eval()
    with torch.no_grad(): 
        fx = input_test_fine[:, idx_closure_inner[:, 1], idx_closure_inner[:, 0]]
        x = fx[:, :, 1:]
        u_pred_inner = model_inner(fx, x_in=x, x_out=x, iphi=model_inner_iphi)
        fx = input_test_fine[:, idx_closure_outer[:, 1], idx_closure_outer[:, 0]]
        x = fx[:, :, 1:]
        u_pred_outer = model_outer(fx, x_in=x, x_out=x, iphi=model_outer_iphi)
        up_pred = torch.concat((u_pred_inner, u_pred_outer), dim=-2).squeeze(-1)
        print('test error on high resolution: relative L2 norm = ', torch.linalg.norm(up_pred.flatten() - u_test_fine.flatten()).item() / torch.linalg.norm(u_test_fine.flatten()).item())
        print('test error on high resolution: relative L_infty norm = ', torch.linalg.norm(up_pred.flatten() -  u_test_fine.flatten(), ord=torch.inf).item() / torch.linalg.norm(u_test_fine.flatten(), ord=torch.inf).item())
    plt.figure()
    plt.plot(np.arange(0, epochs+1, 100), rel_l2_history, '-*')
    plt.xlabel('epochs')
    plt.ylabel('relative l2 error')
    # plt.ylim(1e-3, 1e+2)
    plt.yscale("log")
    plt.savefig('DeepONet-type/2d-L-shaped/saved_data/unsupervised_fno_l2.png')
    up_pred = np.zeros((N*M+1, N*M+1))
    up_pred[idx_closure_inner[:, 1], idx_closure_inner[:, 0]] = np.array(u_pred_inner[0]).squeeze()
    up_pred[idx_closure_outer[:, 1], idx_closure_outer[:, 0]] = np.array(u_pred_outer[0]).squeeze()
    u_test_fine = np.zeros((N*M+1, N*M+1))
    u_test_fine[idx_closure_inner[:, 1], idx_closure_inner[:, 0]] = np.array(u_test_fine_inner[0])
    u_test_fine[idx_closure_outer[:, 1], idx_closure_outer[:, 0]] = np.array(u_test_fine_outer[0])

    grid_fine = np.linspace(0, 1, N*M+1)
    xx,yy = np.meshgrid(grid_fine, grid_fine)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(xx, yy, up_pred, cmap='rainbow')
    ax1.set_title('Predicted Solution u(x,y)')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(xx, yy, u_test_fine, cmap='rainbow')
    ax2.set_title('Reference Solution u(x,y)')
    plt.tight_layout()
    plt.savefig('DeepONet-type/2d-L-shaped/test_pino.png')