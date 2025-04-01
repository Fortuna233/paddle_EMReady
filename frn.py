#Filter Response Normolization
import sys
import torch
import torch.nn as nn

# 声明公开接口1d, 2d, 3d
__all__ = ['FilterResponseNorm1d', 'FilterResponseNorm2d', 'FilterResponseNorm3d']

class FilterResponseNormNd(nn.Module):
    def __init__(self, ndim, num_features, eps=1e-6,
                 learnable_eps=False):
        assert ndim in [3, 4, 5], 'FilterResponseNorm only supports 3d, 4d, 5d inputs.'

        super(FilterResponseNormNd, self).__init__()
        shape = (1, num_features) + (1, ) * (ndim - 2)
        self.eps = nn.Parameter(torch.ones(*shape) * eps)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        
        self.gamma = nn.Parameter(torch.Tensor(*shape))
        self.beta = nn.Parameter(torch.Tensor(*shape))
        self.tau = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    
    def forward(self, x):
        avg_dims = tuple(range(2, x.dim()))
        nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)
    

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)


class FilterResponseNorm1d(FilterResponseNormNd):
    def __init__(self, num_features, eps=0.000001, learnable_eps=False):
        super(FilterResponseNorm1d, self).__init__(3, num_features, eps=eps, learnable_eps=learnable_eps)


class FilterResponseNorm2d(FilterResponseNormNd):
    def __init__(self, num_features, eps=0.000001, learnable_eps=False):
        super(FilterResponseNorm2d, self).__init__(4, num_features, eps=eps, learnable_eps=learnable_eps)


class FilterResponseNorm3d(FilterResponseNormNd):
    def __init__(self, num_features, eps=0.000001, learnable_eps=False):
        super(FilterResponseNorm3d, self).__init__(5, num_features, eps=eps, learnable_eps=learnable_eps)


if __name__ == "__main__":
    x = torch.ones((64, 32, 48, 48, 48))
    frn1 = FilterResponseNorm3d(32)
    frn1(x) 