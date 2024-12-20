

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections import OrderedDict
from scipy import interpolate
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
    if x < 1/2:
        a = 16
    else:
        a = 1
    return a 

def b(x,y):
    return torch.where(x>=1/2, 2*(1-x), 0.0)

N = 32
M = 4  # M-times test-resolution
type_ = 'unsupervised' # unsupervised
ntrain = 1000 
ntest = 200
ntotal = ntrain + ntest
alpha = 1 #interface jump
beta = 0
eps = 1.000

epochs = 20000
learning_rate = 0.001
batch_size = 250
step_size = 2000
gamma = 0.6
model = DeepONet(N**2,2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

k = 0 
x = np.linspace(1/(2*N),1-1/(2*N),N)
xx,yy = np.meshgrid(x,x)
gridvec = np.hstack((xx.reshape(N**2,1),yy.reshape(N**2,1)))

data = np.load("DeepONet-type/2d-smooth/saved_data/data.npz")
f_total = np.load('DeepONet-type/2d-smooth/saved_data/f_centor.npy')
up_total = data['up_total']
ut_fine = data['u_test_fine']
ut_fine = torch.tensor(ut_fine, dtype=torch.float32).to(device)

f_total = torch.tensor(f_total.reshape(ntotal, -1), dtype=torch.float32).to(device)
loc_total = np.tile(np.expand_dims(gridvec, axis=0), (ntotal, 1, 1))
loc_total = torch.tensor(loc_total, dtype=torch.float32).to(device)
up_total = torch.tensor(up_total.reshape(ntotal, -1), dtype=torch.float32).to(device)
f_train = f_total[0:ntrain]
loc_train = loc_total[0:ntrain]
loc_test = loc_total[ntrain:]
up_train = up_total[0:ntrain]
up_test = up_total[ntrain:]
f_test = f_total[ntrain:ntotal]
xx_fine, yy_fine = np.meshgrid(np.linspace(0, 1, N*M+1), np.linspace(0, 1, N*M+1))
gridvec = np.hstack((xx_fine.reshape(-1, 1), yy_fine.reshape(-1, 1)))
loc_fine = np.tile(np.expand_dims(gridvec, axis=0), (ntest, 1, 1))
loc_fine = torch.tensor(loc_fine, dtype=torch.float32).to(device)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_train,loc_train,up_train), batch_size=batch_size, shuffle=True)
mseloss = torch.nn.MSELoss(reduction='mean')
mse_history = []
rel_l2_history = []

x = torch.tensor(x, dtype=torch.float32).to(device)
x_b = torch.cat((torch.tensor(list(zip(x, torch.zeros_like(x)))), 
                  torch.tensor(list(zip(x, torch.ones_like(x)))),
                  torch.tensor(list(zip(torch.zeros_like(x), x))),
                  torch.tensor(list(zip(torch.ones_like(x), x)))), axis=0)
y_b = b(x_b[:, 0], x_b[:, 1]).unsqueeze(0).unsqueeze(-1).repeat([batch_size,1,1]).to(device)
x_b = x_b.unsqueeze(0).repeat([batch_size,1,1]).to(device)

xr_i = torch.tensor(list(zip(0.51*torch.ones_like(x), x))).unsqueeze(0).repeat([batch_size,1,1]).to(device)
xl_i = torch.tensor(list(zip(0.49*torch.ones_like(x), x))).unsqueeze(0).repeat([batch_size,1,1]).to(device)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    for x, l, y in train_loader:
        optimizer.zero_grad()
        x1 = l[:,:,0:1]
        x2 = l[:,:,1:2]
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        y_pred_f = model(x,torch.cat((x1, x2), dim=2))
        if type_ == 'supervised':
            mse  = 1000.0*mseloss(y_pred_f.flatten(), y.flatten())
        else:
            y_x1 = torch.autograd.grad(y_pred_f, x1, grad_outputs=torch.ones_like(y_pred_f), create_graph=True)[0]
            y_x1x1 = torch.autograd.grad(y_x1, x1, grad_outputs=torch.ones_like(y_x1), create_graph=True)[0]
            y_x2 = torch.autograd.grad(y_pred_f, x2, grad_outputs=torch.ones_like(y_pred_f), create_graph=True)[0]
            y_x2x2 = torch.autograd.grad(y_x2, x2, grad_outputs=torch.ones_like(y_x2), create_graph=True)[0]
            Cx1x2 = torch.where(x1<=0.5, 16.0, 1.0)
            F = - (y_x1x1 + y_x2x2) + Cx1x2*y_pred_f - x.unsqueeze(2)
            mse_f = torch.mean(F ** 2)
            
            y_bp = model(x,x_b)
            mse_b = mseloss(y_b,y_bp)
            
            yr = model(x,xr_i)
            yl = model(x,xl_i)
            mse_i = mseloss(yr,yl+1)
            
            mse = mse_f + 50.0*mse_b + 50.0*mse_i
        mse.backward()
        optimizer.step()
        train_mse += mse.item()
    scheduler.step()
    t2 = default_timer()
    train_mse /= len(train_loader)
    mse_history.append(train_mse)
    if ep==0 or (ep + 1)%100 ==0:
        up_pred = model(f_test, loc_test)
        rel_l2 = torch.linalg.norm(up_pred.flatten() - up_test.flatten()).item() / torch.linalg.norm(up_test.flatten()).item()
        rel_l2_history.append(rel_l2)
        if type_=='unsupervised':
            print(mse_f.item(), 50.0*mse_b.item(), 50.0*mse_i.item())
        print('epoch {:d}/{:d} , MSE = {:.6f}, relative L2 norm = {:.6f}, using {:.6f}s\n'.format(ep + 1, epochs, train_mse, rel_l2, t2 - t1), end='', flush=True)
np.save('DeepONet-type/2d-smooth/saved_data/{}_deeponet_loss_history.npy'.format(type_), mse_history)
np.save('DeepONet-type/2d-smooth/saved_data/{}_deeponet_rel_l2_history.npy'.format(type_), rel_l2_history)
torch.save(model.state_dict(), 'DeepONet-type/2d-smooth/saved_data/{}_deeponet_model_state.pt'.format(type_))

with torch.no_grad(): 
    up_pred = model(f_test, loc_fine)
    print(torch.linalg.norm(ut_fine.flatten()).item())
    print('test error on high resolution: relative L2 norm = ', torch.linalg.norm(up_pred.flatten() - ut_fine.flatten()).item() / torch.linalg.norm(ut_fine.flatten()).item())
    print('test error on high resolution: relative L_infty norm = ', torch.linalg.norm(up_pred.flatten() -  ut_fine.flatten(), ord=torch.inf).item() / torch.linalg.norm(ut_fine.flatten(), ord=torch.inf).item())
plt.figure()
plt.plot(np.arange(0, epochs+1, 100), rel_l2_history, '-*')
plt.xlabel('epochs')
plt.ylabel('relative l2 error')
# plt.ylim(1e-3, 1e+2)
plt.yscale("log")
plt.savefig('DeepONet-type/2d-smooth/saved_data/{}_deeponet_l2.png'.format(type_))
plt.show()
