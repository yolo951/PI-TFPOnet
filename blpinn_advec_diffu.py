import torch
import numpy as np
#import foamFileOperation
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
#from torchvision import datasets, transforms
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time
import math

#Solve 1D linear steady advection-diffusion eqn with Vel and Diff given:
#Perturbation method (Kutz note/arxiv)

def geo_train(device,x_in,y_in,xb,cb,batchsize,learning_rate,epochs,path,Flag_batch,C_analytical,Vel,Diff,Flag_BC_exact ):
	if (Flag_batch):
		dataset = TensorDataset(torch.Tensor(x_in))
		dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = True )
	else:
		x = torch.Tensor(x_in)  
		y = torch.Tensor(y_in)  
	h_n = 60 # 40
	input_n = 1 # this is what our answer is a function of. In the original example 3 : x,y,scale 
	class Swish(nn.Module):
		def __init__(self, inplace=True):
			super(Swish, self).__init__()
			self.inplace = inplace

		def forward(self, x):
			if self.inplace:
				x.mul_(torch.sigmoid(x))
				return x
			else:
				return x * torch.sigmoid(x)
	class Net3(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net3, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			if (Flag_BC_exact):
				output = output*x*(x-1) + (-0.9*x + 1.) #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
				#output = output*x*(1-x) + torch.exp(math.log(0.1)*x) #Do it exponentially? Not as good
			return  output

	################################################################
	net_original = Net3().to(device)  #original method
	###### Initialize the neural network using a standard method ##############
	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	# use the modules apply function to recursively apply the initialization
	net_original.apply(init_normal)
	############################################################
	optimizer3 = optim.Adam(net_original.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	###### Definte the PDE and physics loss here ##############
	def criterion_original(x):

		#print (x)
		x = torch.Tensor(x).to(device)
		#x = torch.FloatTensor(x).to(device)
		#x= torch.from_numpy(x).to(device)
		x.requires_grad = True
		#net_in = torch.cat((x),1)
		net_in = x
		C = net_original(net_in)
		C = C.view(len(C),-1)
		c_x = torch.autograd.grad(C,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		c_xx = torch.autograd.grad(c_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		
		loss_1 =   Diff * c_xx + (1 + Diff) * c_x + C

		# MSE LOSS
		loss_f = nn.MSELoss()
		#Note our target is zero. It is residual so we use zeros_like
		loss = loss_f(loss_1,torch.zeros_like(loss_1)) 

		return loss
	###### Define boundary conditions ##############
	###################################################################
	def Loss_BC_original(xb,cb):
		xb = torch.FloatTensor(xb).to(device)
		cb = torch.FloatTensor(cb).to(device)
		#xb.requires_grad = True
		#net_in = torch.cat((xb),1)
		out = net_original(xb)
		cNN = out.view(len(out), -1)
		#cNN = cNN*(1.-xb) + cb    #cNN*xb*(1-xb) + cb
		loss_f = nn.MSELoss()
		loss_bc = loss_f(cNN, cb/U_scale)
		return loss_bc

	######## Main loop ###########
	tic = time.time()
	for epoch in range(epochs):

		net_original.zero_grad()
		loss_eqn_original = criterion_original(x) #the original method (traditional adv-dif solver)
		loss_bc_original = Loss_BC_original(xb,cb)
		loss_original = loss_eqn_original + Lambda_bc*loss_bc_original
		loss_original.backward()

		optimizer3.step() 
		if epoch % 5 ==0:
			print('Loss original {:.10f} Loss_eqn_original {:.10f} Loss_bc_original: {:.8f}'.format( loss_original.item() ,loss_eqn_original.item(),loss_bc_original.item()  ))

	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time = ", elapseTime)
	###################
	results = net_original(x)  #evaluate model
	C_Result_original = U_scale * results.data.numpy()
	#### Plot ########
	plt.figure()
	plt.plot(x.detach().numpy(), C_analytical[:], '-', label='Analytical solution', alpha=1.0,zorder=0) #analytical
	plt.plot(x.detach().numpy() , C_Result_original, 'k-', label='Original PINN', alpha=1.,zorder=0) #PINN
	plt.legend(loc='best')
	plt.show()

	return

#Main code:
device = torch.device("cpu")
epochs =  2000  #5000 

Flag_batch = False #Use batch or not 
Flag_Chebyshev = False #Use Chebyshev pts for more accurcy in BL region
Flag_BC_exact = False #If True enforces BC exactly HELPS ALOT here!!!
Flag_pretrain = False #IF true read previous files

Lambda_bc = 1.

## Parameters###
Vel = 1.0
Diff = 0.005 / 10.

nPt = 100 
xStart = 0.
xEnd = 1.

if(Flag_Chebyshev): #!!!Not a very good idea (makes even the simpler case worse)
	x = np.polynomial.chebyshev.chebpts1(2*nPt)
	x = x[nPt:]
	if(0):#Mannually place more pts at the BL 
		x = np.linspace(0.95, xEnd, nPt)
		x[1] = 0.2
		x[2] = 0.5
	x[0] = 0.
	x[-1] = xEnd
	x = np.reshape(x, (nPt,1))
else:
	x = np.linspace(xStart, xEnd, nPt)
	x = np.reshape(x, (nPt,1))

#zeta = x/Diff
#y = zeta / inf_scale   ( 0<y<1)

y = np.linspace(0, 1, nPt)
y = np.reshape(y, (nPt,1))
print('shape of x',x.shape)
#boundary pt and boundary condition
#X_BC_loc = 1.
#C_BC = 1.
#xb = np.array([X_BC_loc],dtype=np.float32)
#cb = np.array([C_BC ], dtype=np.float32)
C_BC1 = 0.
C_BC2 = 1.
xb = np.array([0.,1.],dtype=np.float32)
cb = np.array([C_BC1,C_BC2], dtype=np.float32)
xb= xb.reshape(-1, 1) #need to reshape to get 2D array
cb= cb.reshape(-1, 1) #need to reshape to get 2D array
#xb = np.transpose(xb)  #transpose because of the order that NN expects instances of training data
#cb = np.transpose(cb)

batchsize = 50 #50
learning_rate = 1e-4

inf_scale = 10. #5   #used to set BC at infinity.  FINAL DEFAULT VALUE FOR THIS CODE: 10

U_scale = math.exp(1.) # This is the max value in analytical soln; need to scale by this
path = "nips/"
#Analytical soln
A = (np.exp(-x[:]) -  np.exp(-x[:]/Diff))
B = 1. / ( exp(-1.) -  exp(-1./Diff))
C_analytical = A * B

#path = pre+"aneurysmsigma01scalepara_100pt-tmp_"+str(ii)
net2_final = geo_train(device,x,y,xb,cb,batchsize,learning_rate,epochs,path,Flag_batch,C_analytical,Vel,Diff,Flag_BC_exact )
#tic = time.time()