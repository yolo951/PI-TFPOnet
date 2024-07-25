
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from timeit import default_timer
from scipy import interpolate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepONet(nn.Module):
    def __init__(self, b_dim, t_dim):
        super(DeepONet, self).__init__()
        self.b_dim = b_dim
        self.t_dim = t_dim

        self.branch = nn.Sequential(
            nn.Linear(self.b_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.t_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

        self.b = Parameter(torch.zeros(1))

    def forward(self, x, l):
        x = self.branch(x)
        l = self.trunk(l)

        res = torch.einsum("bi,bi->b", x, l)
        res = res.unsqueeze(1) + self.b
        return res


ntrain, ntest = 1000, 200
epochs = 3000
type_ = 'unsupervised' # 'unsupervised'
q1 = lambda x: 2*x+1
q2 = lambda x: 2*(1-x)+1
q = lambda x: torch.where(x<=0.5, q1(x), q2(x))

f = np.load('f.npy')
u1 = np.load('u1.npy')
u2 = np.load('u2.npy')

# u = np.concatenate((u1, u2[:, 1:]), axis=-1)

Nx = f.shape[-1]
half_Nx = int(Nx/2)+1
f0 = f[:ntrain, :half_Nx]
f1 = f[:ntrain, half_Nx-1:]
# batch_size = 2 ** 8 + 1  # dim


N = ntrain * half_Nx
grid = np.linspace(0, 1, Nx)
grid0, grid1 = grid[:half_Nx], grid[half_Nx-1:]

input_loc0 = np.tile(grid0, ntrain).reshape((N, 1))
input_loc1 = np.tile(grid1, ntrain).reshape((N, 1))
input_f0 = np.repeat(f0, half_Nx, axis=0)
input_f1 = np.repeat(f1, half_Nx, axis=0)
output0 = u1[:ntrain].reshape((N, 1))
output1 = u2[:ntrain].reshape((N, 1))
input_f0 = torch.Tensor(input_f0).to(device)
input_f1 = torch.Tensor(input_f1).to(device)
input_loc0 = torch.Tensor(input_loc0).to(device)
input_loc1 = torch.Tensor(input_loc1).to(device)
output0 = torch.Tensor(output0).to(device)
output1 = torch.Tensor(output1).to(device)
rhs0 = torch.tensor(f0).reshape((N, 1)).float().to(device)
rhs1 = torch.tensor(f1).reshape((N, 1)).float().to(device)
loc_b0 = torch.tile(torch.tensor([0.]), (N, 1)).to(device)
loc_mid = torch.tile(torch.tensor([0.5]), (N, 1)).to(device)
loc_b1 = torch.tile(torch.tensor([1.]), (N, 1)).to(device)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_f0, input_loc0, output0, rhs0, loc_b0, input_f1, input_loc1, output1, rhs1, loc_b1, loc_mid),
                                           batch_size=1024, shuffle=True)

model0 = DeepONet(half_Nx,  1).to(device)
model1 = DeepONet(half_Nx,  1).to(device)

optimizer0 = torch.optim.Adam(model0.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler0 = torch.optim.lr_scheduler.StepLR(optimizer0, step_size=500, gamma=0.5)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=500, gamma=0.5)
start = default_timer()
for ep in range(epochs):
    model0.train()
    model1.train()
    t1 = default_timer()
    train_mse = 0
    for x0, l0, y0, r0, lb0, x1, l1, y1, r1, lb1, lb_mid in train_loader:
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        l0.requires_grad = True
        l1.requires_grad = True
        lb_mid.requires_grad = True
        out0 = model0(x0, l0)
        out1 = model1(x1, l1)
        if type_ == 'supervised':
            mse = 1000.0*F.mse_loss(out0.view(out0.numel(), -1), y0.view(y0.numel(), -1), reduction='mean')\
                + 1000.0*F.mse_loss(out1.view(out1.numel(), -1), y1.view(y1.numel(), -1), reduction='mean')
        else:
            y_l0 = torch.autograd.grad(outputs=out0, inputs=l0, grad_outputs=torch.ones_like(out0), create_graph=True)[0]
            y_l1 = torch.autograd.grad(outputs=out1, inputs=l1, grad_outputs=torch.ones_like(out1), create_graph=True)[0]
            y_ll0 = torch.autograd.grad(outputs=y_l0, inputs=l0, grad_outputs=torch.ones_like(y_l0), create_graph=True)[0]
            y_ll1 = torch.autograd.grad(outputs=y_l1, inputs=l1, grad_outputs=torch.ones_like(y_l1), create_graph=True)[0]
            mse_equ = 20.0*F.mse_loss(-0.001*y_ll0+q1(l0)*out0, r0) + 20.0*F.mse_loss(-y_ll1+q2(l1)*out1, r1)
            out0, out1 = model0(x0, lb0), model1(x1, lb1)
            mse_b = 100.0*F.mse_loss(out0, torch.zeros_like(out0)) + 100.0*F.mse_loss(out1, torch.zeros_like(out1))
            out0, out1 = model0(x0, lb_mid), model1(x1, lb_mid)
            y_l0_mid = torch.autograd.grad(outputs=out0, inputs=lb_mid, grad_outputs=torch.ones_like(out0), create_graph=True)[0]
            y_l1_mid = torch.autograd.grad(outputs=out1, inputs=lb_mid, grad_outputs=torch.ones_like(out1), create_graph=True)[0]
            mse_i = F.mse_loss(out1-out0, torch.ones_like(out0)) + F.mse_loss(y_l1_mid-y_l0_mid, torch.ones_like(y_l0_mid))
            mse = mse_equ + mse_b + mse_i
        mse.backward()
        optimizer0.step()
        optimizer1.step()
        train_mse += mse.item()
    scheduler0.step()
    scheduler1.step()
    train_mse /= len(train_loader)
    t2 = default_timer()
    print('\repoch {:d}/{:d} , MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse, t2 - t1), end='',
            flush=True)

print('Total training time:', default_timer() - start, 's')
N_fine = 257
half_N_fine = int(N_fine/2)+1
N = ntest * half_N_fine
f_test0 = f[-ntest:, :half_Nx]
f_test1 = f[-ntest:, half_Nx-1:]
grid_fine = np.linspace(0, 1, N_fine)
grid0 = grid_fine[:half_N_fine]
grid1 = grid_fine[half_N_fine-1:]
u_test = np.load('u_test.npy')
input_f0 = np.repeat(f_test0, half_N_fine, axis=0)
input_f1 = np.repeat(f_test1, half_N_fine, axis=0)
input_loc0 = np.tile(grid0, ntest).reshape((N, 1))
input_loc1 = np.tile(grid1, ntest).reshape((N, 1))
input_f0 = torch.Tensor(input_f0).to(device)
input_f1 = torch.Tensor(input_f1).to(device)
input_loc0 = torch.Tensor(input_loc0).to(device)
input_loc1 = torch.Tensor(input_loc1).to(device)




index = 0
test_mse = 0
with torch.no_grad():
    out0 = model0(input_f0, input_loc0).reshape((ntest,half_N_fine))
    out1 = model1(input_f1, input_loc1).reshape((ntest,half_N_fine))
    pred = torch.concat((out0, out1[:, 1:]), dim=-1)
    pred = pred.detach().cpu().numpy()
    fig = plt.figure(figsize=(4, 3), dpi=150)
    plt.plot(grid_fine[:int(N_fine/2+1)], u_test[-ntest, :int(N_fine/2+1)].flatten(), 'b-', label='Ground Truth', linewidth=2, alpha=1., zorder=0)
    plt.plot(grid_fine[int(N_fine/2+1):], u_test[-ntest, int(N_fine/2+1):].flatten(), 'b-', linewidth=2, alpha=1., zorder=0)
    plt.plot(grid_fine[:int(N_fine/2+1)], pred[-ntest, :int(N_fine/2+1)], 'r--', label='Prediction', linewidth=2, alpha=1., zorder=0)
    plt.plot(grid_fine[int(N_fine/2+1):], pred[-ntest, int(N_fine/2+1):], 'r--', linewidth=2, alpha=1., zorder=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('1d_smooth_example')
    print('test error on high resolution: relative L2 norm = ', np.linalg.norm(pred-u_test) / np.linalg.norm(u_test))
    print('test error on high resolution: relative L_inf norm = ', np.linalg.norm(pred-u_test, ord=np.inf) / np.linalg.norm(u_test))
    





