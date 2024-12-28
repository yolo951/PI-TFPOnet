
import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
from scipy import interpolate
from dim2_cnn import encoder_decoder
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        
        self.depth = len(layers) - 1
        self.activation = torch.nn.ReLU
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)
    
    def input_encoding(self, x):
        w = 2.0 * torch.pi / 1
        return torch.hstack([torch.ones_like(x).to(device), torch.cos(1 * w * x), torch.cos(2 * w * x), torch.cos(4 * w * x),  torch.sin(1 * w * x), torch.sin(2 * w * x), torch.sin(4 * w * x)])
    
    def forward(self, x):
        # x = self.input_encoding(x)
        out = self.layers(x)
        return out

def c(x,y):
    if (0.25<=y<=0.5 and 0.25<=x<=0.75) or (0.5<=y<=0.75 and 0.25<=x<=0.5):
        a = 1
    else:
        a = 16
    return a*1000.0

def b(x,y):
    if y==1:
        a = 1-x
    elif y==0:
        a = x
    elif x==1:
        a = 1-y
    else:
        a = y
    return a*0.5

def generate_mask(N):
    # idx except interface
    idx_y = [[N//4+i]*(N//2-1) for i in range(1, N//4)] + [[N//2+i]*(N//4-1) for i in range(0, N//4)]
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
ntrain = 1000  
ntest = 200
ntotal = ntrain + ntest
alpha = 1 #interface jump
beta = 0
eps = 1.0   # We multiply both sides of the equation by 1/eps, so eps here can be 1.0

epochs = 10000
learning_rate = 0.001
batch_size = 64
step_size = 1000
gamma = 0.6
model = DNN([(N+1)**2,512,256,256,128,256,256,512,4*N**2]).to(device)
# model = encoder_decoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# U_total = np.load(r'DeepONet-type/2d-L-shaped/matrixU.npy')
data = np.load("DeepONet-type/2d-L-shaped/saved_data/data.npz")
f_total = data['f_total']/1000.0
f = f_total.reshape((ntotal, N+1, N+1))
B_total, C_total, up_total, index_of_u, val_of_u, ut_fine = data['B_total'], data['C_total'], data['up_total'], data['index'], data['val'], data['u_test_fine']
f_train = torch.tensor(f_total[0:ntrain], dtype=torch.float32).to(device)
index_of_u = torch.LongTensor(index_of_u).to(device)
val_of_u = torch.tensor(val_of_u, dtype=torch.float32).to(device)
B_train = torch.tensor(B_total[0:ntrain], dtype=torch.float32).to(device)
C_train = torch.tensor(C_total[0:ntrain], dtype=torch.float32).to(device)
up_train = up_total[0:ntrain]
# U_train = torch.tensor(U_total[0:ntrain], dtype=torch.float32).to(device)
f_test = torch.tensor(f_total[ntrain:ntotal], dtype=torch.float32).to(device)
up_test = torch.tensor(up_total[ntrain:ntotal], dtype=torch.float32).to(device)
C_test = torch.tensor(C_total[ntrain:ntotal], dtype=torch.float32).to(device)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_train, B_train, C_train), batch_size=batch_size, shuffle=True)
mseloss = torch.nn.MSELoss()

index_jump = []
index_continuous = []
index_boundary = []
for i in range(4*N):
    index_boundary.append(i)
for i in range(0,N-1):
    for j in range(0,N):
        if i==int(N/2)-1: 
            index_jump.append(4*N+N*i+j)
            index_jump.append(4*N+N*(N-1)+N*i+j)
        else:
            index_continuous.append(4*N+N*i+j)
            index_continuous.append(4*N+N*(N-1)+N*i+j)
for i in range(0,N):
    for j in range(0,N-1):
        index_continuous.append(4*N+2*N*(N-1)+N*j+i)
        index_continuous.append(4*N+3*N*(N-1)+N*j+i)
index_jump = torch.LongTensor(index_jump).to(device)
index_continuous = torch.LongTensor(index_continuous).to(device)
index_boundary = torch.LongTensor(index_boundary).to(device)
loss_history = []
rel_l2_history = []
for i in range(epochs):
    model.train()
    train_mse = 0
    for fb, Bb, Cb in train_loader:
        optimizer.zero_grad()
        Cb_pred = model(fb)
        val_u = val_of_u.unsqueeze(0).repeat(fb.shape[0], 1, 1)
        index_u = index_of_u.unsqueeze(0).repeat(fb.shape[0], 1, 1)
        U_jump = torch.index_select(val_u, 1, index_jump)
        B_jump = torch.index_select(Bb, 1, index_jump)
        U_continuous = torch.index_select(val_u, 1, index_continuous)
        B_continuous = torch.index_select(Bb, 1, index_continuous)
        U_boundary = torch.index_select(val_u, 1, index_boundary)
        B_boundary = torch.index_select(Bb, 1, index_boundary)
        new_Cb_pred = torch.gather(Cb_pred.unsqueeze(-1).expand(-1, -1, 8), 1, index_u)
        loss_jump = mseloss(torch.einsum('bri, bri->br', U_jump, torch.index_select(new_Cb_pred, 1, index_jump)), B_jump)
        loss_continuous = mseloss(torch.einsum('bri, bri->br', U_continuous, torch.index_select(new_Cb_pred, 1, index_continuous)), B_continuous)
        loss_boundary = mseloss(torch.einsum('bri, bri->br', U_boundary, torch.index_select(new_Cb_pred, 1, index_boundary)), B_boundary)
        loss = 100.0*loss_jump + 100.0*loss_continuous +100.0*loss_boundary
        loss.backward()                  
        optimizer.step()  
        train_mse += loss.item()
    scheduler.step()
    train_mse /= len(train_loader)
    loss_history.append(train_mse)
    
    if i==0 or (i+1)%100==0:
        # print(100.0*loss_jump.item(), 100.0*loss_continuous.item(), 100.0*loss_boundary.item())
        u_pred = model(f_test).reshape(ntest, -1, 4).sum(axis=-1)
        u = C_test.reshape(ntest, -1, 4).sum(axis=-1)
        rel_l2 = torch.linalg.norm(u_pred.flatten() - u.flatten()) / torch.linalg.norm(u.flatten())
        rel_l_infty = torch.linalg.norm(u_pred.flatten() - u.flatten(), ord=torch.inf) / torch.linalg.norm(u.flatten(), ord=torch.inf)
        rel_l2_history.append(rel_l2.item())
        print('epoch',i,': loss ',train_mse, 'rel_l2 ',rel_l2.item(), 'rel_l_infty ',rel_l_infty.item())
np.save('DeepONet-type/2d-L-shaped/saved_data/loss_history.npy', loss_history)
np.save('DeepONet-type/2d-L-shaped/saved_data/rel_l2_history.npy', rel_l2_history)
torch.save(model.state_dict(), 'DeepONet-type/2d-L-shaped/saved_data/model_state.pt')

C_pred = model(f_train).detach().cpu().reshape(f_train.shape[0], -1)
up_pred = np.zeros((ntrain,N,N))
for k in range(ntrain):
    interpolate_f_2d = interpolate.RegularGridInterpolator((np.linspace(0, 1, N+1),np.linspace(0, 1, N+1)), f[k])
    F = lambda x, y : interpolate_f_2d((x,y))*1000.0
    C = C_pred[k]    
    for i in range(0,N):
        for j in range(0,N):
            x0 = (2*i+1)/(2*N)
            y0 = (2*j+1)/(2*N)
            f0 = F(x0,y0)
            c0 = c(x0,y0)
            mu0 = np.sqrt(c0)/eps
            c1 = C[4*N*j+4*i]
            c2 = C[4*N*j+4*i+1]
            c3 = C[4*N*j+4*i+2]
            c4 = C[4*N*j+4*i+3]
            up_pred[k,j,i] = f0/c0 + c1 + c2 + c3 + c4
rel_l2 = np.linalg.norm(up_pred - up_train) / np.linalg.norm(up_train)
print('relative l2 error on train data: ',rel_l2)     

C_pred = model(f_test).detach().cpu().reshape(f_test.shape[0], -1)
up_pred = np.zeros((ntest,N,N))

M = 4 # M-times test-resolution
up_refine = np.zeros((ntest, M*N+1,M*N+1))
for k in range(ntest):
    interpolate_f_2d = interpolate.RegularGridInterpolator((np.linspace(0, 1, N+1),np.linspace(0, 1, N+1)), f[ntrain+k])
    F = lambda x, y : interpolate_f_2d((x,y))*1000.0
    C = C_pred[k]    
    for i in range(0,N):
        for j in range(0,N):
            x0 = (2*i+1)/(2*N)
            y0 = (2*j+1)/(2*N)
            f0 = F(x0,y0)
            c0 = c(x0,y0)
            mu0 = np.sqrt(c0)/eps
            c1 = C[4*N*j+4*i]
            c2 = C[4*N*j+4*i+1]
            c3 = C[4*N*j+4*i+2]
            c4 = C[4*N*j+4*i+3]
            up_pred[k,j,i] = f0/c0 + c1 + c2 + c3 + c4
    Cp = C_pred[k]
    Ct = C_total[ntrain+k]
    hh = 1/(M*N)
    for i in range(0,N):
        for j in range(0,N):
            x0 = (2*i+1)/(2*N)
            y0 = (2*j+1)/(2*N)
            f0 = F(x0,y0)
            c0 = c(x0,y0)
            mu0 = np.sqrt(c0)/eps
            c1p = Cp[4*N*j+4*i]
            c2p = Cp[4*N*j+4*i+1]
            c3p = Cp[4*N*j+4*i+2]
            c4p = Cp[4*N*j+4*i+3]
            for ki in range(0,M):
                for kj in range(0,M):
                    xhi = -1/(2*N) + ki*hh
                    xhj = -1/(2*N) + kj*hh
                    up_refine[k, j*M+kj,i*M+ki] = f0/c0 + c1p*np.exp(mu0*xhi) + c2p*np.exp(-mu0*xhi) + c3p*np.exp(mu0*xhj) + c4p*np.exp(-mu0*xhj)
    for l in range(0,M*N+1):
        s = l*hh
        up_refine[k, 0,l] = b(s,0)
        up_refine[k, M*N,l] = b(s,1)
        up_refine[k, l,0] = b(0,s)
        up_refine[k, l,M*N] = b(1,s)

k = random.randrange(ntest) # select a random sample from the test dataset to show the error between the true value and the predicted value
up_test = np.array(up_test.cpu())
x = np.linspace(1/(2*N),1-1/(2*N),N)
xx,yy = np.meshgrid(x,x)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xx, yy, up_pred[k], cmap='rainbow')
ax.title.set_text('predicted solution u(x,y)')
# plt.show()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xx, yy, up_test[k], cmap='rainbow')
ax.title.set_text('reference solution u(x,y)')
# plt.show()
rel_l2 = np.linalg.norm(up_pred - up_test) / np.linalg.norm(up_test)
print('relative l2 error on test data: ',rel_l2)

rel_l2 = np.linalg.norm(up_refine - ut_fine) / np.linalg.norm(ut_fine)
rel_l_infty = np.linalg.norm((up_refine - ut_fine).flatten(), ord=np.inf) / np.linalg.norm(ut_fine.flatten(), ord=np.inf)
print('relative l2 error on test data (M-times test-resolution): ',rel_l2)
print('relative l_infty error on test data (M-times test-resolution): ',rel_l_infty)

xh = np.linspace(0,1,N*M+1)
yh = np.linspace(0,1,N*M+1)
xxh, yyh = np.meshgrid(xh, yh)
mask = generate_mask(N*M)
idx_x, idx_y, idx_x_remain, idx_y_remain = mask['idx_x'], mask['idx_y'], mask['idx_x_remain'], mask['idx_y_remain']

Z_refine_inner = np.full_like(xxh, np.nan, dtype=float)
Z_refine_outer = np.full_like(xxh, np.nan, dtype=float)
Z_fine_inner = np.full_like(xxh, np.nan, dtype=float)
Z_fine_outer = np.full_like(xxh, np.nan, dtype=float)
Z_error_inner = np.full_like(xxh, np.nan, dtype=float)
Z_error_outer = np.full_like(xxh, np.nan, dtype=float)
Z_refine_inner[idx_x, idx_y] = up_refine[idx_x, idx_y]
Z_refine_inner[idx_x_remain, idx_y_remain] = up_refine[idx_x_remain, idx_y_remain]
Z_fine_inner[idx_x, idx_y] = ut_fine[idx_x, idx_y]
Z_fine_inner[idx_x_remain, idx_y_remain] = ut_fine[idx_x_remain, idx_y_remain]
error = np.abs(ut_fine-up_refine)
Z_error_inner[idx_x, idx_y] = error[idx_x, idx_y]
Z_error_outer[idx_x_remain, idx_y_remain] = error[idx_x_remain, idx_y_remain]

fig = plt.figure(figsize=(12, 3.5))
# [left, bottom, width, height]
ax0 = fig.add_axes([0.05, 0.1, 0.25, 0.8])
ax1 = fig.add_axes([0.34, 0.1, 0.25, 0.8])
ax_cb = fig.add_axes([0.60, 0.1, 0.01, 0.8])
ax2 = fig.add_axes([0.68, 0.1, 0.25, 0.8])
ax_cb2 = fig.add_axes([0.94, 0.1, 0.01, 0.8])

vmin = min(up_refine.min(), ut_fine.min())
vmax = max(up_refine.max(), ut_fine.max())
levels = np.linspace(vmin, vmax, 100)
cs0 = ax0.contourf(xxh, yyh, Z_refine_inner, levels=levels, cmap='RdYlBu_r')
ax0.contourf(xxh, yyh, Z_refine_outer, levels=levels, cmap='RdYlBu_r')
cs1 = ax1.contourf(xxh, yyh, Z_fine_inner, levels=levels, cmap='RdYlBu_r')
ax1.contourf(xxh, yyh, Z_fine_outer, levels=levels, cmap='RdYlBu_r')
cbar = fig.colorbar(cs0, cax=ax_cb, format='%.3f')
levels_error = np.linspace(np.min(np.concatenate([error[idx_x, idx_y], error[idx_x_remain, idx_y_remain]])),
                            np.max(np.concatenate([error[idx_x, idx_y], error[idx_x_remain, idx_y_remain]]))+1e-3, 100)
cs2 = ax2.contourf(xxh, yyh, Z_error_inner, levels=levels_error, cmap='RdYlBu_r')
ax2.contourf(xxh, yyh, Z_error_outer, levels=levels_error, cmap='RdYlBu_r')
cbar2 = fig.colorbar(cs2, cax=ax_cb2, format='%.3f')

ax0.set_title('Refinement prediction', fontsize=14)
ax1.set_title('Ground Truth', fontsize=14)
ax2.set_title('Point-wise error', fontsize=14)

for ax in [ax0, ax1, ax2]:
    ax.set_aspect('equal')
plt.savefig('DeepONet-type/2d-L-shaped/saved_data/2d_Lshaped_compare.png')

plt.figure()
plt.plot(loss_history)
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.ylim(1e-5, 1e+2)
plt.yscale("log")
plt.savefig('DeepONet-type/2d-L-shaped/saved_data/2d_Lshaped_loss.png')
# plt.show()

plt.figure()
plt.plot(np.arange(0, epochs+1, 100), rel_l2_history, '-*')
plt.xlabel('epochs')
plt.ylabel('relative l2 error')
# plt.ylim(1e-3, 1e+2)
plt.yscale("log")
plt.savefig('DeepONet-type/2d-L-shaped/saved_data/2d_Lshaped_l2.png')
plt.show()
