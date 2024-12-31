import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from timeit import default_timer
from Adam import Adam
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
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w4 = nn.Conv2d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        #x = F.pad(x, [0,self.padding, 0,self.padding])

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
        x = F.gelu(x)

        x1 = self.conv4(x)
        x2 = self.w4(x)
        x = x1 + x2

        #x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

N = 32
M = 4 # M-times test-resolution
ntrain = 1000
ntest = 200
ntotal = ntrain+ntest
batch_size = 50
learning_rate = 0.001
epochs = 10000
step_size = 1500
gamma = 0.5
modes1 = 12
modes2 = 12
width = 16

data = np.load("DeepONet-type/2d-smooth/saved_data/data.npz")
data_fno = np.load("DeepONet-type/2d-smooth/saved_data/data_fno.npz")
x_data = data['f_total'].reshape((ntotal, N+1, N+1))
x_train = torch.tensor(x_data[:ntrain], dtype=torch.float32).unsqueeze(-1).to(device)
y_train = torch.tensor(data_fno['u_train_sparse'], dtype=torch.float32).to(device)

grid_fine = np.linspace(0, 1, N*M+1)
X, Y = np.meshgrid(grid_fine, grid_fine)
points = np.stack((Y.flatten(), X.flatten()), axis=-1)
f_fine_test = torch.tensor(data_fno['f_test_fine'], dtype=torch.float32).unsqueeze(-1).to(device)  # test f on fine grid
ut_fine = torch.tensor(data['u_test_fine'], dtype=torch.float32).to(device)  # test u on fine grid
x_test = torch.tensor(x_data[-ntest:], dtype=torch.float32).unsqueeze(-1).to(device) # test f on sparse grid
y_test = ut_fine[:, ::M, ::M]  # test u on sparse grid

model = FNO2d(modes1, modes2, width).to(device)
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
myloss = torch.nn.MSELoss(reduction='mean')
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
start = default_timer()

mse_history = []
rel_l2_history = []

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        mse = 1000.0*myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        mse.backward()
        optimizer.step()
        train_mse += mse.item()

    scheduler.step()
    train_mse /= len(train_loader)
    mse_history.append(train_mse)
    t2 = default_timer()
    if ep==0 or (ep+1)%100==0:
        up_pred = model(x_test)
        rel_l2 = torch.linalg.norm(up_pred.flatten() - y_test.flatten()).item() / torch.linalg.norm(y_test.flatten()).item()
        rel_l2_history.append(rel_l2)
        print('\repoch {:d}/{:d} L2 = {:.6f}, MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, rel_l2, train_mse, t2 - t1), end='\n', flush=True)
np.save('DeepONet-type/2d-smooth/saved_data/supervised_fno_loss_history.npy', mse_history)
np.save('DeepONet-type/2d-smooth/saved_data/supervised_fno_rel_l2_history.npy', rel_l2_history)
torch.save(model.state_dict(), 'DeepONet-type/2d-smooth/saved_data/supervised_fno_model_state.pt')

with torch.no_grad(): 
    up_pred = model(f_fine_test)
    print('test error on high resolution: relative L2 norm = ', torch.linalg.norm(up_pred.flatten() - ut_fine.flatten()).item() / torch.linalg.norm(ut_fine.flatten()).item())
    print('test error on high resolution: relative L_infty norm = ', torch.linalg.norm(up_pred.flatten() -  ut_fine.flatten(), ord=torch.inf).item() / torch.linalg.norm(ut_fine.flatten(), ord=torch.inf).item())
plt.figure()
plt.plot(np.arange(0, epochs+1, 100), rel_l2_history, '-*')
plt.xlabel('epochs')
plt.ylabel('relative l2 error')
# plt.ylim(1e-3, 1e+2)
plt.yscale("log")
plt.savefig('DeepONet-type/2d-smooth/saved_data/supervised_fno_l2.png')
plt.show()