
import torch
import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp
import matplotlib.pyplot as plt

# generate random functions(1d) default dim=1001
def generate(samples=1000, begin=0, end=1, random_dim=101, out_dim=1001, length_scale=1, interp="cubic", A=0):
    space = GRF(begin, end, length_scale=length_scale, N=random_dim, interp=interp)
    features = space.random(samples, A)
    x_grid = np.linspace(begin, end, out_dim)
    x_data = space.eval_u(features, x_grid[:, None])
    return x_data  # X_data.shape=(samples,out_dim)，每一行表示一个GRF在x_grid上的取值，共有samples个GRF

# Gaussian Random Field
class GRF(object):
    def __init__(self, begin=0, end=1, kernel="RBF", length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(begin, end, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, n, A):
        """Generate `n` random feature vectors.
        """
        u = np.random.randn(self.N, n)
        return np.dot(self.L, u).T + A

    def eval_u_one(self, y, x):
        """Compute the function value at `x` for the feature `y`.
        """
        if self.interp == "linear":
            return np.interp(x, np.ravel(self.x), y)
        f = interpolate.interp1d(
            np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        )
        return f(x)

    def eval_u(self, ys, sensors):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """
        if self.interp == "linear":
            return np.vstack([np.interp(sensors, np.ravel(self.x), y).T for y in ys])

        res = np.zeros((ys.shape[0], sensors.shape[0]))
        for i in range(ys.shape[0]):
            res[i, :] = interpolate.interp1d(np.ravel(self.x), ys[i], kind=self.interp, copy=False, assume_sorted=True)(
                sensors).T
        return res

def weighted_mse_loss(y_pred, y_true, weights):
    # 自定义加权MSE损失函数
    squared_errors = torch.square(y_pred - y_true)
    weighted_errors = squared_errors * weights
    return torch.sum(weighted_errors)


# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        if y.size()[-1] == 1:
            eps = 0.00001
        else:
            eps = 0
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / (y_norms + eps))
            else:
                return torch.sum(diff_norms / (y_norms + eps))

        return diff_norms / (y_norms + eps)

    def __call__(self, x, y):
        return self.rel(x, y)
