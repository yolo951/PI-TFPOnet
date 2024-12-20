
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
gamma = 0.6
model0 = DeepONet(half_N*N,2).to(device)
model1 = DeepONet(half_N*N,2).to(device)
optimizer0 = torch.optim.Adam(model0.parameters(), lr=learning_rate, weight_decay=1e-4)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler0 = torch.optim.lr_scheduler.StepLR(optimizer0, step_size=step_size, gamma=gamma)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=step_size, gamma=gamma)

data = np.load("DeepONet-type/2d-smooth/saved_data/data.npz")
f_total = np.load('DeepONet-type/2d-smooth/saved_data/f_centor.npy').reshape((ntotal, N, N))
up_total = data['up_total']
ut_fine = torch.tensor(data['u_test_fine'], dtype=torch.float32).to(device)

f0 = f_total[:, :, :half_N]
f1 = f_total[:, :, half_N:]
x = np.linspace(1/(2*N),1-1/(2*N),N)
x0, x1 = x[:half_N], x[half_N:]
xx, yy = np.meshgrid(x0, x)
loc0 =  np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
loc0 = np.tile(np.expand_dims(loc0,axis=0),(ntotal,1,1))
xx, yy = np.meshgrid(x1, x)
loc1 = np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
loc1 = np.tile(np.expand_dims(loc1,axis=0),(ntotal,1,1))
u0 = up_total[:, :, :half_N]
u1 = up_total[:, :, half_N:]
up_test = torch.tensor(up_total[ntrain:], dtype=torch.float32).to(device)

x_fine = np.linspace(0, 1, 1+N*M)
x0, x1 = x_fine[:N*M//2], x_fine[N*M//2:]
xx, yy = np.meshgrid(x0, x_fine)
input_loc_test0 =  np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
input_loc_test0 = np.tile(np.expand_dims(input_loc_test0,axis=0),(ntest,1,1))
input_loc_test0 = torch.tensor(input_loc_test0, dtype=torch.float32).to(device)
xx, yy = np.meshgrid(x1, x_fine)
input_loc_test1 =  np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
input_loc_test1 = np.tile(np.expand_dims(input_loc_test1,axis=0),(ntest,1,1))
input_loc_test1 = torch.tensor(input_loc_test1, dtype=torch.float32).to(device)
f_test0 = torch.tensor(f0[-ntest:].reshape(ntest,-1), dtype=torch.float32).to(device)
f_test1 = torch.tensor(f1[-ntest:].reshape(ntest,-1), dtype=torch.float32).to(device)

f_train0 = torch.tensor(f0[:ntrain].reshape(ntrain,-1), dtype=torch.float32).to(device)
f_train1 = torch.tensor(f1[:ntrain].reshape(ntrain,-1), dtype=torch.float32).to(device)
input_loc0 = torch.tensor(loc0[:ntrain], dtype=torch.float32).to(device)
input_loc1 = torch.tensor(loc1[:ntrain], dtype=torch.float32).to(device)
u_train0 = torch.tensor(u0[:ntrain].reshape(ntrain,-1), dtype=torch.float32).to(device)
u_train1 = torch.tensor(u1[:ntrain].reshape(ntrain,-1), dtype=torch.float32).to(device)
 
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_train0,input_loc0,u_train0, f_train1,input_loc1,u_train1), batch_size=batch_size, shuffle=True)
mseloss = torch.nn.MSELoss(reduction='mean')
mse_history = []
rel_l2_history = []
input_loc0 = torch.tensor(loc0[ntrain:], dtype=torch.float32).to(device)
input_loc1 = torch.tensor(loc1[ntrain:], dtype=torch.float32).to(device)

x = torch.tensor(x, dtype=torch.float32).to(device)
x_b0 = torch.cat((torch.tensor(list(zip(x[:half_N], torch.zeros_like(x[:half_N])))), 
                  torch.tensor(list(zip(x[:half_N], torch.ones_like(x[:half_N])))),
                  torch.tensor(list(zip(torch.zeros_like(x), x)))), axis=0)
x_b1 = torch.cat((torch.tensor(list(zip(x[half_N:], torch.zeros_like(x[half_N:])))), 
                  torch.tensor(list(zip(x[half_N:], torch.ones_like(x[half_N:])))),
                  torch.tensor(list(zip(torch.ones_like(x), x)))), axis=0)
y_b0 = b(x_b0[:, 0], x_b0[:, 1]).unsqueeze(0).unsqueeze(-1).repeat([batch_size,1,1]).to(device)
y_b1 = b(x_b1[:, 0], x_b1[:, 1]).unsqueeze(0).unsqueeze(-1).repeat([batch_size,1,1]).to(device)
x_b0 = x_b0.unsqueeze(0).repeat([batch_size,1,1]).to(device)
x_b1 = x_b1.unsqueeze(0).repeat([batch_size,1,1]).to(device)

x_i = torch.tensor(list(zip(0.5*torch.ones_like(x), x))).unsqueeze(0).repeat([batch_size,1,1]).to(device)

for ep in range(epochs):
    model0.train()
    model1.train()
    t1 = default_timer()
    train_mse = 0
    for ff0, l0, y0, ff1, l1, y1 in train_loader:
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        x00 = l0[:,:,0:1]
        x01 = l0[:,:,1:2]
        x10 = l1[:,:,0:1]
        x11 = l1[:,:,1:2]
        x00.requires_grad_(True)
        x01.requires_grad_(True)
        x10.requires_grad_(True)
        x11.requires_grad_(True)
        y_pred0 = model0(ff0,torch.cat((x00, x01), dim=2))
        y_pred1 = model1(ff1,torch.cat((x10, x11), dim=2))
        if type_ == 'supervised':
            mse0 = 100.0*mseloss(y_pred0.flatten(), y0.flatten())
            mse1 = 100.0*mseloss(y_pred1.flatten(), y1.flatten())
        else:
            y_x1 = torch.autograd.grad(y_pred0, x00, grad_outputs=torch.ones_like(y_pred0), create_graph=True)[0]
            y_x1x1 = torch.autograd.grad(y_x1, x00, grad_outputs=torch.ones_like(y_x1), create_graph=True)[0]
            y_x2 = torch.autograd.grad(y_pred0, x01, grad_outputs=torch.ones_like(y_pred0), create_graph=True)[0]
            y_x2x2 = torch.autograd.grad(y_x2, x01, grad_outputs=torch.ones_like(y_x2), create_graph=True)[0]
            # Cx1x2 = torch.gt(x00,0.5)*16+(~torch.gt(x00,0.5))*1
            Cx1x2 = torch.where(x00<=0.5, 16.0, 1.0)
            F = - 0.001*(y_x1x1 + y_x2x2) + Cx1x2*y_pred0 - ff0.unsqueeze(2)
            mse_f0 = torch.mean(F ** 2)
            
            y_x1 = torch.autograd.grad(y_pred1, x10, grad_outputs=torch.ones_like(y_pred1), create_graph=True)[0]
            y_x1x1 = torch.autograd.grad(y_x1, x10, grad_outputs=torch.ones_like(y_x1), create_graph=True)[0]
            y_x2 = torch.autograd.grad(y_pred1, x11, grad_outputs=torch.ones_like(y_pred1), create_graph=True)[0]
            y_x2x2 = torch.autograd.grad(y_x2, x11, grad_outputs=torch.ones_like(y_x2), create_graph=True)[0]
            # Cx1x2 = torch.gt(x10,0.5)*16+(~torch.gt(x10,0.5))*1
            Cx1x2 = torch.where(x10<=0.5, 16.0, 1.0)
            F = - (y_x1x1 + y_x2x2) + Cx1x2*y_pred1 - ff1.unsqueeze(2)
            mse_f1 = torch.mean(F ** 2)

            y_bp0 = model0(ff0, x_b0)
            mse_b0 = mseloss(y_b0, y_bp0)
            y_bp1 = model1(ff1,x_b1)
            mse_b1 = mseloss(y_b1, y_bp1)
            
            yr = model1(ff1, x_i)
            yl = model0(ff0, x_i)
            mse_i = mseloss(yr,yl+1)
            mse0 = 10*mse_f0 + 10*mse_b0 + 10*mse_i
            mse1 = 10*mse_f1 + 10*mse_b1
        mse0.backward()
        mse1.backward()
        optimizer0.step()
        optimizer1.step()
        train_mse += mse0.item()+mse1.item()
    scheduler0.step()
    scheduler1.step()
    train_mse /= len(train_loader)
    t2 = default_timer()
    mse_history.append(train_mse)
    if ep==0 or (ep + 1)%100 ==0:
        out0 = model0(f_test0, input_loc0).reshape(ntest, N, half_N)
        out1 = model1(f_test1, input_loc1).reshape(ntest, N, half_N)
        pred = torch.concat((out0, out1), dim=-1)
        rel_l2 = torch.linalg.norm(pred.flatten() - up_test.flatten()).item() / torch.linalg.norm(up_test.flatten()).item()
        rel_l2_history.append(rel_l2)
        if type_=='unsupervised':
            print(10*(mse_f0.item() + mse_f0.item()),10*(mse_b0.item()+mse_b1.item()), mse_i.item())
        print('epoch {:d}/{:d} , MSE = {:.6f}, rel_l2 = {:.6f}, using {:.6f}s\n'.format(ep + 1, epochs, train_mse, rel_l2, t2 - t1), end='', flush=True)
np.save('DeepONet-type/2d-smooth/saved_data/{}_ionet_loss_history.npy'.format(type_), mse_history)
np.save('DeepONet-type/2d-smooth/saved_data/{}_ionet_rel_l2_history.npy'.format(type_), rel_l2_history)
torch.save({'model0': model0.state_dict(), 'model1': model1.state_dict()}, 'DeepONet-type/2d-smooth/saved_data/{}_ionet_model_state.pth'.format(type_))

with torch.no_grad():
    out0 = model0(f_test0, input_loc_test0).reshape(ntest, N*M+1, N*M//2)
    out1 = model1(f_test1, input_loc_test1).reshape(ntest, N*M+1, N*M//2+1)
    pred = torch.concat((out0, out1), dim=-1)
    print('test error on high resolution: relative L2 norm = ', torch.linalg.norm(pred.flatten() - ut_fine.flatten()).item() / torch.linalg.norm(ut_fine.flatten()).item())
    print('test error on high resolution: relative L_infty norm = ', torch.linalg.norm(pred.flatten() -  ut_fine.flatten(), ord=torch.inf).item() / torch.linalg.norm(ut_fine.flatten(), ord=torch.inf).item())
plt.figure()
plt.plot(np.arange(0, epochs+1, 100), rel_l2_history, '-*')
plt.xlabel('epochs')
plt.ylabel('relative l2 error')
# plt.ylim(1e-3, 1e+2)
plt.yscale("log")
plt.savefig('DeepONet-type/2d-smooth/saved_data/{}_ionet_l2.png'.format(type_))
plt.show()

