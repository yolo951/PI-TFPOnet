
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
            nn.Linear(self.b_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.t_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
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
q1 = lambda x: 1+torch.exp(x)
q2 = lambda x: 1-torch.log(x+1)
q = lambda x: torch.where(x<=0.5, q1(x), q2(x))

f = np.load('f.npy')
u1 = np.load('u1.npy')
u2 = np.load('u2.npy')
u = np.concatenate((u1, u2[:, 1:]), axis=-1)

Nx = f.shape[-1]

batch_size = 2 ** 8 + 1  # dim


N = ntrain * Nx
grid = np.linspace(0, 1, Nx)

input_loc = np.tile(grid, ntrain).reshape((N, 1))
input_f = np.repeat(f[:ntrain], Nx, axis=0)
output = u[:ntrain].reshape((N, 1))
input_f = torch.Tensor(input_f).to(device)
input_loc = torch.Tensor(input_loc).to(device)
output = torch.Tensor(output).to(device)
rhs = torch.tensor(f[:ntrain]).reshape((N, 1)).float().to(device)
loc_b0 = torch.tile(torch.tensor([0.]), (N, 1)).to(device)
loc_b1 = torch.tile(torch.tensor([1.]), (N, 1)).to(device)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_f, input_loc, output, rhs, loc_b0, loc_b1),
                                           batch_size=256, shuffle=True)

model = DeepONet(Nx,  1).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
start = default_timer()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    for x, l, y, r, lb0, lb1 in train_loader:
        optimizer.zero_grad()
        l.requires_grad = True
        out = model(x, l)
        if type_ == 'supervised':
            mse = 1000.0*F.mse_loss(out.view(out.numel(), -1), y.view(y.numel(), -1), reduction='mean')
        else:
            y_l = torch.autograd.grad(outputs=out, inputs=l, grad_outputs=torch.ones_like(out), create_graph=True)[0]
            y_ll = torch.autograd.grad(outputs=y_l, inputs=l, grad_outputs=torch.ones_like(y_l), create_graph=True)[0]
            mse = 50.0*F.mse_loss(-y_ll+q(l)*out, r)
            out0, out1 = model(x, lb0), model(x, lb1)
            mse += 5.0*F.mse_loss(out0, torch.zeros_like(out0))+2*F.mse_loss(out1, torch.zeros_like(out1))

        mse.backward()
        optimizer.step()
        train_mse += mse.item()
    scheduler.step()
    train_mse /= len(train_loader)
    t2 = default_timer()
    print('\repoch {:d}/{:d} , MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse, t2 - t1), end='',
            flush=True)

print('Total training time:', default_timer() - start, 's')
N_fine = 257
N = ntest * N_fine
f_test = f[-ntest:, :]
grid_fine = np.linspace(0, 1, N_fine)
u_test = np.load('u_test.npy')
input_f = np.repeat(f_test, N_fine, axis=0)
input_loc = np.tile(grid_fine, ntest).reshape((N, 1))
output = u_test.reshape((N, 1))
input_f = torch.Tensor(input_f).to(device)
input_loc = torch.Tensor(input_loc).to(device)
output = torch.Tensor(output).to(device)



index = 0
test_mse = 0
with torch.no_grad():
    out = model(input_f, input_loc).reshape((ntest, N_fine))
    pred = out.detach().cpu().numpy()
    fig = plt.figure(figsize=(4, 3), dpi=150)
    plt.plot(grid_fine[:int(N_fine/2+1)], u_test[-ntest, :int(N_fine/2+1)].flatten(), 'b-', label='Ground Truth', linewidth=2, alpha=1., zorder=0)
    plt.plot(grid_fine[int(N_fine/2+1):], u_test[-ntest, int(N_fine/2+1):].flatten(), 'b-', linewidth=2, alpha=1., zorder=0)
    plt.plot(grid_fine[:int(N_fine/2+1)], pred[-ntest, :int(N_fine/2+1)], 'r--', label='Prediction', linewidth=2, alpha=1., zorder=0)
    plt.plot(grid_fine[int(N_fine/2+1):], pred[-ntest, int(N_fine/2+1):], 'r--', linewidth=2, alpha=1., zorder=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('1d_smooth_example')
    print('test error on high resolution: MSE = ', np.linalg.norm(pred-u_test) / np.linalg.norm(u_test))
    





