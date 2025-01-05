
import sys
sys.path.append('DeepONet-type')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from Adam import Adam
from scipy import integrate
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, begin=0, end=1, input_num=1):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.begin = begin
        self.end = end
        self.modes1 = modes
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(input_num + 1, self.width)  # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(self.begin, self.end, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

x_data = np.load('DeepONet-type/1d-singular/f.npy')
y1 = np.load('DeepONet-type/1d-singular/u1.npy')
y2 = np.load('DeepONet-type/1d-singular/u2.npy')
y_data = np.hstack((y1[:, :-1], y2))
x_grid = np.linspace(0, 1, x_data.shape[-1])

ntrain = 1000
ntest = 200

batch_size = 50
learning_rate = 0.001

epochs = 3000
step_size = 500
gamma = 0.5

modes = 16
width = 64
x_train = x_data[:ntrain]
y_train = y_data[:ntrain]
x_train = torch.Tensor(x_train).unsqueeze(-1).to(device)
y_train = torch.Tensor(y_train).unsqueeze(-1).to(device)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
model = FNO1d(modes, width).cuda()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

start = default_timer()
myloss = torch.nn.MSELoss()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)
        mse = 100.*myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        mse.backward()

        optimizer.step()
        train_mse += mse.item()

    scheduler.step()
    train_mse /= len(train_loader)
    train_l2 /= ntrain
    t2 = default_timer()
    print('\repoch {:d}/{:d} , MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse,
                                                                              t2 - t1), end='\n', flush=True)

print('Total training time:', default_timer() - start, 's')

N_fine = 257
grid_fine = np.linspace(0, 1, N_fine)
x_test = np.load('DeepONet-type/1d-singular/f_test_fine.npy')
y_test = np.load('DeepONet-type/1d-singular/u_test.npy')
x_test = torch.Tensor(x_test).unsqueeze(-1).to(device)

index = 0
test_mse = 0
with torch.no_grad():
    out = model(x_test).reshape((ntest, N_fine))
    pred = out.detach().cpu().numpy()
    fig = plt.figure(figsize=(4, 3), dpi=150)
    plt.plot(grid_fine[:int(N_fine/2+1)], y_test[-ntest, :int(N_fine/2+1)].flatten(), 'b-', label='Ground Truth', linewidth=2, alpha=1., zorder=0)
    plt.plot(grid_fine[int(N_fine/2+1):], y_test[-ntest, int(N_fine/2+1):].flatten(), 'b-', linewidth=2, alpha=1., zorder=0)
    plt.plot(grid_fine[:int(N_fine/2+1)], pred[-ntest, :int(N_fine/2+1)], 'r--', label='Prediction', linewidth=2, alpha=1., zorder=0)
    plt.plot(grid_fine[int(N_fine/2+1):], pred[-ntest, int(N_fine/2+1):], 'r--', linewidth=2, alpha=1., zorder=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('DeepONet-type/1d-singular/1d_fno_example.png')
    print('test error on high resolution: relative L2 norm = ', np.linalg.norm(pred-y_test) / np.linalg.norm(y_test))
    print('test error on high resolution: relative L_inf norm = ', np.linalg.norm(pred-y_test, ord=np.inf) / np.linalg.norm(y_test))