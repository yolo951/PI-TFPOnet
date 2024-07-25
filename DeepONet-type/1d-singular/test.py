
from train import DNN, PhysicsInformedNN
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import interpolate
import dill
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntrain = 500
ntest = 50
factor = 10
NS = 64
f1 = np.load('ex2/ex2_f1.npy')
f2 = np.load('ex2/ex2_f2.npy')
u1 = np.load('ex2/ex2_u1.npy')
u2 = np.load('ex2/ex2_u2.npy')
u1 *= factor
u2 *= factor
N2_max = f1.shape[-1]
x_h = np.linspace(-1, 1, 2*N2_max-1)
y_h = np.linspace(-1, 1, 2*N2_max-1)

N_f = N = 33
layers = [N_f, 64, 64, 64, 64, 2*N]
model = DNN(layers).to(device)
model.load_state_dict(torch.load('model.pt'))

def predict(grid_f, grid_coarse, x, f_A, f_B, f_rhs, f_AB, F):
    model.eval()
    pred_u = np.zeros((len(F), len(x)))
    for k in range(len(F)):
        input_ = torch.tensor(F[k](grid_f)/1000).float().to(device)
        pred_AB = model(input_).reshape((-1, 2))
        pred_AB = pred_AB.detach().cpu().numpy()
        
        A_x, B_x = [], []
        A_x_pred, B_x_pred = [], []
        def f_AB_pred(x):
            result = pred_AB[0]
            for i in range(1, N):
                result = np.where((grid_coarse[i-1] < x) & (x <= grid_coarse[i]), pred_AB[i], result)
            return result
        for i in range(len(x)):
            pred_u[k, i] = f_AB_pred(x[i])[0]*f_A(x[i])+f_AB_pred(x[i])[1]*f_B(x[i])+f_rhs[k](x[i])
            A_x.append(f_AB[k](x[i])[0])
            A_x_pred.append(f_AB_pred(x[i])[0])
            B_x.append(f_AB[k](x[i])[1])
            B_x_pred.append(f_AB_pred(x[i])[1])
    return pred_u


f = np.load('f.npy')
interpolate_f = interpolate.interp1d(np.linspace(0, 1, f.shape[-1]), f)
F = [lambda x, k=k: interpolate_f(x)[k] for k in range(f.shape[0])]


N_f = 33
grid_f = np.linspace(0, 1, N_f)
u1 = np.load('u1.npy')
u2 = np.load('u2.npy')
U = np.load('U.npy')
B = np.load('B.npy')
with open('f_AB.pkl', 'rb') as ff:
    f_AB = dill.load(ff)
with open('f_A.pkl', 'rb') as ff:
    f_A = dill.load(ff)
with open('f_B.pkl', 'rb') as ff:
    f_B = dill.load(ff)
with open('f_rhs.pkl', 'rb') as ff:
    f_rhs = dill.load(ff)

