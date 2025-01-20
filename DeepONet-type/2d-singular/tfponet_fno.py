
import sys
sys.path.append('/root/PI-TFPOnet/DeepONet-type')
from Adam import Adam
import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import OrderedDict
from scipy import interpolate
from dim2_cnn import encoder_decoder
import random
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv4 = SpectralConv2d(self.width, self.width, setttttlf.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        # self.w4 = nn.Conv2d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        #x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = nn.functional.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = nn.functional.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = nn.functional.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        # x = nn.functional.gelu(x)

        # x1 = self.conv4(x)
        # x2 = self.w4(x)
        # x = x1 + x2

        #x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = nn.functional.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(1/size_x, 1-1/size_x, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(1/size_y, 1-1/size_y, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

def c(x,y):
    if x < 1/2:
        a = 16
    else:
        a = 1
    return a*1000.0

def b(x,y):
    if x >= 1/2:
        a = 1-x
    else:
        a = -x
    return a

N = 32
ntrain = 1000  
ntest = 200
ntotal = ntrain + ntest
alpha = 1 #interface jump
beta = 0
eps = 1.0   # We multiply both sides of the equation by 1/eps, so eps here can be 1.0

epochs = 5000
learning_rate = 0.001
batch_size = 32
step_size = 1000
gamma = 0.5
modes1 = 12
modes2 = 12
width = 32
model = FNO2d(modes1, modes2, width).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"# params: {total_params}")
# model = encoder_decoder().to(device)
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# U_total = np.load(r'DeepONet-type/2d-singular/matrixU.npy')
data = np.load("DeepONet-type/2d-singular/saved_data/data.npz")
f_centor = np.load('DeepONet-type/2d-singular/saved_data/f_centor.npy').reshape((ntotal, N, N))
f_total = data['f_total']/1000.0
f = f_total.reshape((ntotal, N+1, N+1))
B_total, C_total, up_total, index_of_u, val_of_u, ut_fine = data['B_total'], data['C_total'], data['up_total'], data['index'], data['val'], data['u_test_fine']
f_train = torch.tensor(f_centor[0:ntrain], dtype=torch.float32).unsqueeze(-1).to(device)
index_of_u = torch.LongTensor(index_of_u).to(device)
val_of_u = torch.tensor(val_of_u, dtype=torch.float32).to(device)
B_train = torch.tensor(B_total[0:ntrain], dtype=torch.float32).to(device)
C_train = torch.tensor(C_total[0:ntrain], dtype=torch.float32).to(device)
up_train = up_total[0:ntrain]
# U_train = torch.tensor(U_total[0:ntrain], dtype=torch.float32).to(device)
f_test = torch.tensor(f_centor[ntrain:ntotal], dtype=torch.float32).unsqueeze(-1).to(device)
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
    t1 = time.time()
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
        new_Cb_pred = torch.gather(Cb_pred.reshape((fb.shape[0], -1)).unsqueeze(-1).expand(-1, -1, 8), 1, index_u)
        loss_jump = mseloss(torch.einsum('bri, bri->br', U_jump, torch.index_select(new_Cb_pred, 1, index_jump)), B_jump)
        loss_continuous = mseloss(torch.einsum('bri, bri->br', U_continuous, torch.index_select(new_Cb_pred, 1, index_continuous)), B_continuous)
        loss_boundary = mseloss(torch.einsum('bri, bri->br', U_boundary, torch.index_select(new_Cb_pred, 1, index_boundary)), B_boundary)
        loss = 50.0*loss_jump + 50.0*loss_continuous +100.0*loss_boundary
        loss.backward()                  
        optimizer.step()  
        train_mse += loss.item()
    scheduler.step()
    train_mse /= len(train_loader)
    loss_history.append(train_mse)
    t2 = time.time()
    if i==0 or (i+1)%100==0:
        print(t2-t1)
        u_pred = model(f_test).reshape(ntest, -1, 4).sum(axis=-1)
        u = C_test.reshape(ntest, -1, 4).sum(axis=-1)
        rel_l2 = torch.linalg.norm(u_pred.flatten() - u.flatten()) / torch.linalg.norm(u.flatten())
        rel_l_infty = torch.linalg.norm(u_pred.flatten() - u.flatten(), ord=torch.inf) / torch.linalg.norm(u.flatten(), ord=torch.inf)
        rel_l2_history.append(rel_l2.item())
        print('epoch',i,': loss ',train_mse, 'rel_l2 ',rel_l2.item(), 'rel_l_infty ',rel_l_infty.item())
np.save('DeepONet-type/2d-singular/saved_data/tfpo_fno_loss.npy', loss_history)
np.save('DeepONet-type/2d-singular/saved_data/tfpo_fno_rel_l2.npy', rel_l2_history)
torch.save(model.state_dict(), 'DeepONet-type/2d-singular/saved_data/tfpo_fno_model.pt')

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
xxh1, yyh1 = np.meshgrid(xh[:N*M//2], yh)
xxh2, yyh2 = np.meshgrid(xh[N*M//2+1:], yh)

fig = plt.figure(figsize=(12, 3.5))
# [left, bottom, width, height]
ax0 = fig.add_axes([0.05, 0.1, 0.25, 0.8])
ax1 = fig.add_axes([0.34, 0.1, 0.25, 0.8])
ax_cb = fig.add_axes([0.60, 0.1, 0.01, 0.8])
ax2 = fig.add_axes([0.68, 0.1, 0.25, 0.8])
ax_cb2 = fig.add_axes([0.94, 0.1, 0.01, 0.8])

vmin = min(up_refine[k].min(), ut_fine[k].min())
vmax = max(up_refine[k].max(), ut_fine[k].max())
levels = np.linspace(vmin, vmax, 100)
cs0 = ax0.contourf(xxh1, yyh1, up_refine[k, :, :N*M//2], levels=levels, cmap='RdYlBu_r')
ax0.contourf(xxh2, yyh2, up_refine[k, :, N*M//2+1:], levels=levels, cmap='RdYlBu_r')
cs1 = ax1.contourf(xxh1, yyh1, ut_fine[k, :, :N*M//2], levels=levels, cmap='RdYlBu_r')
ax1.contourf(xxh2, yyh2, ut_fine[k, :, N*M//2+1:], levels=levels, cmap='RdYlBu_r')
cbar = fig.colorbar(cs0, cax=ax_cb, format='%.3f')
error = np.abs(up_refine[k]-ut_fine[k])
error = np.hstack((error[:, :N*M//2], error[:, N*M//2+1:]))
levels_error = np.linspace(error.min(), error.max(), 100)
cs2 = ax2.contourf(xxh1, yyh1, error[:, :N*M//2], levels=levels_error, cmap='RdYlBu_r')
ax2.contourf(xxh2, yyh2, error[:, N*M//2:], levels=levels_error, cmap='RdYlBu_r')
cbar2 = fig.colorbar(cs2, cax=ax_cb2, format='%.3f')

ax0.set_title('Refinement prediction', fontsize=14)
ax1.set_title('Ground Truth', fontsize=14)
ax2.set_title('Point-wise error', fontsize=14)

for ax in [ax0, ax1, ax2]:
    ax.set_aspect('equal')
# plt.savefig('DeepONet-type/2d-singular/saved_data/2d_singular_compare.png')

plt.figure()
plt.plot(loss_history)
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.ylim(1e-5, 1e+2)
plt.yscale("log")
plt.savefig('DeepONet-type/2d-singular/saved_data/tfpo_fno_loss.png')
# plt.show()

plt.figure()
plt.plot(np.arange(0, epochs+1, 100), rel_l2_history, '-*')
plt.xlabel('epochs')
plt.ylabel('relative l2 error')
# plt.ylim(1e-3, 1e+2)
plt.yscale("log")
plt.savefig('DeepONet-type/2d-singular/saved_data/tfpo_fno_l2.png')
plt.show()