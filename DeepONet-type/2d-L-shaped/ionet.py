
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections import OrderedDict
from timeit import default_timer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class DeepONet(nn.Module):
    def __init__(self,b_dim,t_dim):
        super(DeepONet, self).__init__()
        self.b_dim = b_dim
        self.t_dim = t_dim
        
        self.branch = nn.Sequential(
            nn.Linear(self.b_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
        )
        
        self.trunk = nn.Sequential(
            nn.Linear(self.t_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
        )
        
        self.b = Parameter(torch.zeros(1))

        
    def forward(self, x, l):
        x = self.branch(x)
        l = self.trunk(l)
        
        res = torch.einsum("bi,bki->bk", x, l)
        res = res.unsqueeze(-1) + self.b
        return res
    
def c(x,y):
    if (0.25<=y<=0.5 and 0.25<=x<=0.75) or (0.5<=y<=0.75 and 0.25<=x<=0.5):
        a = 1
    else:
        a = 16
    return a*1000.0

def b(x, y):
    a = torch.where(y == 1, 1 - x,
        torch.where(y == 0, x,
        torch.where(x == 1, 1 - y, y)))
    return a * 0.5

def generate_mask(N):
    idx_y = [[N//4+i]*(N//2-1) for i in range(1, N//4)] + [[N//2+i]*(N//4-1) for i in range(0, N//4)]
    # idx except interface
    idx_y_remain = [[i]*(N-1) for i in range(1, N//4)] + [[N//4+i]*(N//2-2) for i in range(0, N//4+1)]\
                    + [[N//2+i]*(N*3//4-2) for i in range(1, N//4+1)] + [[N*3//4+i]*(N-1) for i in range(1, N//4)]
    idx_y = np.concatenate(idx_y)
    idx_y_remain = np.concatenate(idx_y_remain)
    idx_x = [N//4+j for j in range(1, N//2)]*(N//4-1) + [N//4+j for j in range(1, N//4)]*(N//4)
    idx_x_remain = np.hstack((np.array([j for j in range(1, N)]*(N//4-1)),
                            np.concatenate([[j for j in range(1, N//4)]+[j for j in range(N*3//4+1, N)]]*(N//4+1)), 
                            np.concatenate([[j for j in range(1, N//4)]+[j for j in range(N//2+1, N)]]*(N//4)),
                            np.array([j for j in range(1, N)]*(N//4-1))))
    idx_x = np.array(idx_x)
    mask = {'idx_x': idx_x, 'idx_x_remain': idx_x_remain, 'idx_y': idx_y, 'idx_y_remain': idx_y_remain}
    return mask

N = 32
M = 4  # M-times test-resolution
half_N = int(N/2)
ntrain = 1000 
ntest = 200
ntotal = ntrain + ntest
alpha = 1 #interface jump
beta = 0
eps = 1.0


type_ = 'supervised' # supervised
epochs = 20000
learning_rate = 0.001
batch_size = 200
step_size = 2000
gamma = 0.5

data = np.load("DeepONet-type/2d-L-shaped/saved_data/data.npz")
f_total = data['f_total'].reshape(ntotal, N+1, N+1)/1000.
u_total = data['up_total']
ut_fine = torch.tensor(data['u_test_fine'], dtype=torch.float32).to(device)
mask = generate_mask(N)
idx_x, idx_y, idx_x_remain, idx_y_remain = mask['idx_x'], mask['idx_y'], mask['idx_x_remain'], mask['idx_y_remain']
model_inner = DeepONet(idx_x.shape[0], 2).to(device)
model_outer = DeepONet(idx_x_remain.shape[0], 2).to(device)

f_inner = f_total[:, idx_y, idx_y]
f_outer = f_total[:, idx_y_remain, idx_x_remain]
f_train_inner = torch.tensor(f_inner[:ntrain], dtype=torch.float32).to(device)
f_train_outer = torch.tensor(f_outer[:ntrain], dtype=torch.float32).to(device)
f_test_inner = torch.tensor(f_inner[ntrain:], dtype=torch.float32).to(device)
f_test_outer = torch.tensor(f_outer[ntrain:], dtype=torch.float32).to(device)
x = np.linspace(0, 1, N+1)
xx, yy = np.meshgrid(x, x)
loc_sparse_inner = np.stack((xx[idx_y, idx_x],yy[idx_y, idx_x]), axis=-1)
loc_train_inner = torch.tensor(np.tile(np.expand_dims(loc_sparse_inner,axis=0),(ntrain,1,1)), dtype=torch.float32).to(device)
loc_sparse_test_inner = torch.tensor(np.tile(np.expand_dims(loc_sparse_inner,axis=0),(ntest,1,1)), dtype=torch.float32).to(device)
loc_sparse_outer = np.stack((xx[idx_y_remain, idx_x_remain],yy[idx_y_remain, idx_x_remain]), axis=-1)
loc_train_outer = torch.tensor(np.tile(np.expand_dims(loc_sparse_outer,axis=0),(ntrain,1,1)), dtype=torch.float32).to(device)
loc_sparse_test_outer = torch.tensor(np.tile(np.expand_dims(loc_sparse_outer,axis=0),(ntest,1,1)), dtype=torch.float32).to(device)

u_inner = u_total[:, idx_y, idx_x]
u_outer = u_total[:, idx_y_remain, idx_x_remain]
u_train_inner = torch.tensor(u_inner[:ntrain], dtype=torch.float32).to(device)
u_train_outer = torch.tensor(u_outer[:ntrain], dtype=torch.float32).to(device)
u_sparse_test_inner = torch.tensor(u_inner[ntrain:], dtype=torch.float32).to(device)
u_sparse_test_outer = torch.tensor(u_outer[ntrain:], dtype=torch.float32).to(device)

x_fine = np.linspace(0, 1, 1+N*M)
xx, yy = np.meshgrid(x_fine, x_fine)
mask = generate_mask(N*M)
idx_x, idx_y, idx_x_remain, idx_y_remain = mask['idx_x'], mask['idx_y'], mask['idx_x_remain'], mask['idx_y_remain']
loc_fine_test_inner = np.stack((xx[idx_y, idx_x],yy[idx_y, idx_x]), axis=-1)
loc_fine_test_inner = torch.tensor(np.tile(np.expand_dims(loc_fine_test_inner,axis=0),(ntest,1,1)), dtype=torch.float32).to(device)
loc_fine_test_outer = np.stack((xx[idx_y_remain, idx_x_remain],yy[idx_y_remain, idx_x_remain]), axis=-1)
loc_fine_test_outer = torch.tensor(np.tile(np.expand_dims(loc_fine_test_outer,axis=0),(ntest,1,1)), dtype=torch.float32).to(device)
u_test_fine = data['u_test_fine']
u_test_fine_inner = torch.tensor(u_test_fine[:, idx_y, idx_x], dtype=torch.float32).to(device)
u_test_fine_outer = torch.tensor(u_test_fine[:, idx_y_remain, idx_x_remain], dtype=torch.float32).to(device)

x = torch.tensor(x, dtype=torch.float32).to(device)
x_b = torch.cat((torch.tensor(list(zip(x, torch.zeros_like(x)))), 
                  torch.tensor(list(zip(x, torch.ones_like(x)))),
                  torch.tensor(list(zip(torch.zeros_like(x), x))),
                  torch.tensor(list(zip(torch.ones_like(x), x)))), axis=0)
y_b = b(x_b[:, 0], x_b[:, 1]).unsqueeze(0).unsqueeze(-1).repeat([batch_size,1,1]).to(device)
x_b = x_b.unsqueeze(0).repeat([batch_size,1,1]).to(device)

dx = 0.
x_inner = [[x0, 0.25+dx] for x0 in np.linspace(0.25+dx, 0.75-dx, 20, endpoint=True)]\
        +[[0.75-dx, y0] for y0 in np.linspace(0.25+dx, 0.5-dx, 10, endpoint=True)]\
        +[[x0, 0.5-dx] for x0 in np.linspace(0.5+dx, 0.75-dx, 10, endpoint=True)]\
        +[[0.5-dx, y0] for y0 in np.linspace(0.5+dx, 0.75-dx, 10, endpoint=True)]\
        +[[x0, 0.75-dx] for x0 in np.linspace(0.25+dx, 0.5-dx, 10, endpoint=True)]\
        +[[0.25+dx, y0] for y0 in np.linspace(0.25+dx, 0.75-dx, 20, endpoint=True)]
x_outer = [[x0, 0.25-dx] for x0 in np.linspace(0.25+dx, 0.75-dx, 20, endpoint=True)]\
        +[[0.75+dx, y0] for y0 in np.linspace(0.25+dx, 0.5-dx, 10, endpoint=True)]\
        +[[x0, 0.5+dx] for  x0 in np.linspace(0.5+dx, 0.75-dx, 10, endpoint=True)]\
        +[[0.5+dx, y0] for y0 in np.linspace(0.5+dx, 0.75-dx, 10, endpoint=True)]\
        +[[x0, 0.75+dx] for x0 in np.linspace(0.25+dx, 0.5-dx, 10, endpoint=True)]\
        +[[0.25-dx, y0] for y0 in np.linspace(0.25+dx, 0.75-dx, 20, endpoint=True)]
x_inner = torch.tensor(x_inner, dtype=torch.float32).unsqueeze(0).repeat([batch_size,1,1]).to(device)
x_outer = torch.tensor(x_outer, dtype=torch.float32).unsqueeze(0).repeat([batch_size,1,1]).to(device)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_train_inner, loc_train_inner, u_train_inner,
                                                                          f_train_outer, loc_train_outer, u_train_outer),
                                                                          batch_size=batch_size, shuffle=True)
optimizer_inner = torch.optim.Adam(model_inner.parameters(), lr=learning_rate, weight_decay=1e-4)
optimizer_outer = torch.optim.Adam(model_outer.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler_inner = torch.optim.lr_scheduler.StepLR(optimizer_inner, step_size=step_size, gamma=gamma)
scheduler_outer = torch.optim.lr_scheduler.StepLR(optimizer_outer, step_size=step_size, gamma=gamma)
mseloss = torch.nn.MSELoss(reduction='mean')
mse_history = []
rel_l2_history = []

for ep in range(epochs):
    model_inner.train()
    model_outer.train()
    t1 = default_timer()
    train_mse = 0
    for ff_i, l_i, y_i, ff_o, l_o, y_o in train_loader:
        optimizer_inner.zero_grad()
        optimizer_outer.zero_grad()
        xi0 = l_i[:,:,0:1]
        xi1 = l_i[:,:,1:2]
        xo0 = l_o[:,:,0:1]
        xo1 = l_o[:,:,1:2]
        xi0.requires_grad_(True)
        xi1.requires_grad_(True)
        xo0.requires_grad_(True)
        xo1.requires_grad_(True)
        y_pred0 = model_inner(ff_i,torch.cat((xi0, xi1), dim=2))
        y_pred1 = model_outer(ff_o,torch.cat((xo0, xo1), dim=2))
        if type_ == 'supervised':
            mse = 100.0*mseloss(y_pred0.flatten(), y_i.flatten())
            mse += 100.0*mseloss(y_pred1.flatten(), y_o.flatten())
        else:
            y_x1 = torch.autograd.grad(y_pred0, xi0, grad_outputs=torch.ones_like(y_pred0), create_graph=True)[0]
            y_x1x1 = torch.autograd.grad(y_x1, xi0, grad_outputs=torch.ones_like(y_x1), create_graph=True)[0]
            y_x2 = torch.autograd.grad(y_pred0, xi1, grad_outputs=torch.ones_like(y_pred0), create_graph=True)[0]
            y_x2x2 = torch.autograd.grad(y_x2, xi1, grad_outputs=torch.ones_like(y_x2), create_graph=True)[0]
            # Cx1x2 = torch.gt(x00,0.5)*16+(~torch.gt(x00,0.5))*1
            Cx1x2 = torch.where(((0.25 <= xi1) & (xi1 <= 0.5) & (0.25 <= xi0) & (xi0 <= 0.75)) |
                                ((0.5 <= xi1) & (xi1 <= 0.75) & (0.25 <= xi0) & (xi0 <= 0.5)), 1.0, 16.0)
            F = - 0.001*(y_x1x1 + y_x2x2) + Cx1x2*y_pred0 - ff_i.unsqueeze(2)
            mse_f0 = torch.mean(F ** 2)
            
            y_x1 = torch.autograd.grad(y_pred1, xo0, grad_outputs=torch.ones_like(y_pred1), create_graph=True)[0]
            y_x1x1 = torch.autograd.grad(y_x1, xo0, grad_outputs=torch.ones_like(y_x1), create_graph=True)[0]
            y_x2 = torch.autograd.grad(y_pred1, xo1, grad_outputs=torch.ones_like(y_pred1), create_graph=True)[0]
            y_x2x2 = torch.autograd.grad(y_x2, xo1, grad_outputs=torch.ones_like(y_x2), create_graph=True)[0]
            # Cx1x2 = torch.gt(x10,0.5)*16+(~torch.gt(x10,0.5))*1
            Cx1x2 = torch.where(((0.25 <= xo1) & (xo1 <= 0.5) & (0.25 <= xo0) & (xo0 <= 0.75)) |
                                ((0.5 <= xo1) & (xo1 <= 0.75) & (0.25 <= xo0) & (xo0 <= 0.5)), 1.0, 16.0)
            F = - 0.001*(y_x1x1 + y_x2x2) + Cx1x2*y_pred1 - ff_o.unsqueeze(2)
            mse_f1 = torch.mean(F ** 2)

            y_bp = model_outer(ff_o, x_b)
            mse_b = mseloss(y_b, y_bp)
            
            yi = model_inner(ff_i, x_inner)
            yo = model_outer(ff_o, x_outer)
            mse_i = mseloss(yi, yo+1)
            mse = 10*mse_f0 + 10*mse_f1 + 10*mse_b + 10*mse_i
        mse.backward()
        optimizer_inner.step()
        optimizer_outer.step()
        train_mse += mse.item()
    scheduler_inner.step()
    scheduler_outer.step()
    train_mse /= len(train_loader)
    t2 = default_timer()
    mse_history.append(train_mse)
    if ep==0 or (ep + 1)%100 ==0:
        out0 = model_inner(f_test_inner, loc_sparse_test_inner)
        out1 = model_outer(f_test_outer, loc_sparse_test_outer)
        pred = torch.concat((out0, out1), dim=-2)
        up_test = torch.concat((u_sparse_test_inner, u_sparse_test_outer), dim=-1)
        rel_l2 = torch.linalg.norm(pred.flatten() - up_test.flatten()).item() / torch.linalg.norm(up_test.flatten()).item()
        rel_l2_history.append(rel_l2)
        print('epoch {:d}/{:d} , MSE = {:.6f}, rel_l2 = {:.6f}, using {:.6f}s\n'.format(ep + 1, epochs, train_mse, rel_l2, t2 - t1), end='', flush=True)
np.save('DeepONet-type/2d-L-shaped/saved_data/{}_ionet_loss_history.npy'.format(type_), mse_history)
np.save('DeepONet-type/2d-L-shaped/saved_data/{}_ionet_rel_l2_history.npy'.format(type_), rel_l2_history)
torch.save({'model_inner': model_inner.state_dict(), 'model_outer': model_outer.state_dict()}, 'DeepONet-type/2d-L-shaped/saved_data/{}_ionet_model_state.pth'.format(type_))

with torch.no_grad():
    out0 = model_inner(f_test_inner, loc_fine_test_inner)
    out1 = model_outer(f_test_outer, loc_fine_test_outer)
    pred = torch.concat((out0, out1), dim=-2)
    ut_fine = torch.concat((u_test_fine_inner, u_test_fine_outer), dim=-1)
    print('test error on high resolution: relative L2 norm = ', torch.linalg.norm(pred.flatten() - ut_fine.flatten()).item() / torch.linalg.norm(ut_fine.flatten()).item())
    print('test error on high resolution: relative L_infty norm = ', torch.linalg.norm(pred.flatten() -  ut_fine.flatten(), ord=torch.inf).item() / torch.linalg.norm(ut_fine.flatten(), ord=torch.inf).item())
plt.figure()
plt.plot(np.arange(0, epochs+1, 100), rel_l2_history, '-*')
plt.xlabel('epochs')
plt.ylabel('relative l2 error')
# plt.ylim(1e-3, 1e+2)
plt.yscale("log")
plt.savefig('DeepONet-type/2d-L-shaped/saved_data/{}_ionet_l2.png'.format(type_))
plt.show()

xh = np.linspace(0,1,N*M+1)
yh = np.linspace(0,1,N*M+1)
xxh, yyh = np.meshgrid(xh, yh)
mask = generate_mask(N*M)
idx_x, idx_y, idx_x_remain, idx_y_remain = mask['idx_x'], mask['idx_y'], mask['idx_x_remain'], mask['idx_y_remain']

k = 0
Z_refine_inner = np.full_like(xxh, np.nan, dtype=float)
Z_refine_outer = np.full_like(xxh, np.nan, dtype=float)
Z_fine_inner = np.full_like(xxh, np.nan, dtype=float)
Z_fine_outer = np.full_like(xxh, np.nan, dtype=float)
Z_error_inner = np.full_like(xxh, np.nan, dtype=float)
Z_error_outer = np.full_like(xxh, np.nan, dtype=float)
out0 = out0[k].squeeze().detach().cpu().numpy()
out1 = out1[k].squeeze().detach().cpu().numpy()
u_test_fine_inner = u_test_fine_inner[k].cpu().numpy()
u_test_fine_outer = u_test_fine_outer[k].cpu().numpy()
Z_refine_inner[idx_y, idx_x] = out0
Z_refine_outer[idx_y_remain, idx_x_remain] = out1
Z_fine_inner[idx_y, idx_x] = u_test_fine_inner
Z_fine_outer[idx_y_remain, idx_x_remain] = u_test_fine_outer
error_inner = np.abs(out0-u_test_fine_inner)
error_outer = np.abs(out1-u_test_fine_outer)
Z_error_inner[idx_y, idx_x] = error_inner
Z_error_outer[idx_y_remain, idx_x_remain] = error_outer

fig = plt.figure(figsize=(12, 3.5))
# [left, bottom, width, height]
ax0 = fig.add_axes([0.05, 0.1, 0.25, 0.8])
ax1 = fig.add_axes([0.34, 0.1, 0.25, 0.8])
ax_cb = fig.add_axes([0.60, 0.1, 0.01, 0.8])
ax2 = fig.add_axes([0.68, 0.1, 0.25, 0.8])
ax_cb2 = fig.add_axes([0.94, 0.1, 0.01, 0.8])

vmin = min(out0.min(), out1.min(), u_test_fine_inner.min(), u_test_fine_inner.min())
vmax = max(out0.max(), out1.max(), u_test_fine_inner.max(), u_test_fine_inner.max())
levels = np.linspace(vmin, vmax, 100)
cs0 = ax0.contourf(xxh, yyh, Z_refine_inner, levels=levels, cmap='RdYlBu_r')
ax0.contourf(xxh, yyh, Z_refine_outer, levels=levels, cmap='RdYlBu_r')
cs1 = ax1.contourf(xxh, yyh, Z_fine_inner, levels=levels, cmap='RdYlBu_r')
ax1.contourf(xxh, yyh, Z_fine_outer, levels=levels, cmap='RdYlBu_r')
cbar = fig.colorbar(cs0, cax=ax_cb, format='%.3f')
levels_error = np.linspace(np.min([error_inner.min(), error_outer.min()]),
                            np.max([error_inner.max(), error_outer.max()]), num=100)
cs2 = ax2.contourf(xxh, yyh, Z_error_inner, levels=levels_error, cmap='RdYlBu_r')
ax2.contourf(xxh, yyh, Z_error_outer, levels=levels_error, cmap='RdYlBu_r')
cbar2 = fig.colorbar(cs2, cax=ax_cb2, format='%.3f')

ax0.set_title('Refinement prediction', fontsize=14)
ax1.set_title('Ground Truth', fontsize=14)
ax2.set_title('Point-wise error', fontsize=14)

for ax in [ax0, ax1, ax2]:
    ax.set_aspect('equal')
plt.savefig('DeepONet-type/2d-L-shaped/test_ionet.png')

