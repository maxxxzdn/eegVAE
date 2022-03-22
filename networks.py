import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric.nn as gnn
from torch_geometric.utils import dense_to_sparse


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Aggregator(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
    def forward(self, input):
        if self.mode == 'mean':
            return input.mean(1)
        elif self.mode == 'max':
            return input.max(1).values
        elif self.mode == 'none':
            return input.view(input.shape[0],-1)

class UnFlatten(nn.Module):
    def __init__(self, var, size=None):
        super().__init__()
        self.var = var
        self.size = size
    def forward(self, input):
        if self.var == 'z':
            return input.view(input.shape[0]*input.shape[1], -1, self.size)
        elif self.var == 'w':
            return input.view(-1, 16, self.size, self.size)
        
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
class ToSymmetric(nn.Module):
    def forward(self, input):
        input = input.tril(-1)
        return input + input.permute(0,1,3,2)

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=256, drop_rate = 0., *args, **kwargs):
        super().__init__()
        # setup the three linear transformations used
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), 
            nn.Dropout(drop_rate),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1), 
            nn.Dropout(drop_rate),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), 
            nn.Dropout(drop_rate),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), 
            nn.Dropout(drop_rate),
            nn.ReLU(True),
            nn.Conv2d(128, hidden_dim, 4, 1),
            nn.Dropout(drop_rate),
            nn.ReLU(True),
            View((-1, hidden_dim*1*1))
        )

        self.locs = nn.Linear(hidden_dim, z_dim)
        self.scales = nn.Linear(hidden_dim, z_dim)


    def forward(self, x):
        hidden = self.encoder(x)
        return 10*torch.tanh(self.locs(hidden)), torch.clamp(F.softplus(self.scales(hidden)), min=1e-3)
     
        
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=256, drop_rate = 0., *args, **kwargs):
        super().__init__()
        # setup the two linear transformations used
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),  
            View((-1, hidden_dim, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 128, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            ToSymmetric(),
        )

    def forward(self, z):
        m = self.decoder(z)
        return m


class Diagonal(nn.Module):
    def __init__(self, dim):
        super(Diagonal, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))

    def forward(self, x):
        return x * self.weight + self.bias

class Classifier(nn.Module):
    def __init__(self, dim):
        super(Classifier, self).__init__()
        self.dim = dim
        self.diag = Diagonal(self.dim)

    def forward(self, x):
        return self.diag(x)

class CondPrior(nn.Module):
    def __init__(self, dim):
        super(CondPrior, self).__init__()
        self.dim = dim
        self.diag_loc_true = nn.Parameter(torch.zeros(self.dim))
        self.diag_loc_false = nn.Parameter(torch.zeros(self.dim))
        self.diag_scale_true = nn.Parameter(torch.ones(self.dim))
        self.diag_scale_false = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        loc = x * self.diag_loc_true + (1 - x) * self.diag_loc_false
        scale = x * self.diag_scale_true + (1 - x) * self.diag_scale_false
        return 10*torch.tanh(loc), torch.clamp(F.softplus(scale), min=1e-3)
