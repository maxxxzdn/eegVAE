import torch
import torch.nn as nn
import torch.nn.functional as F

           
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class Diagonal(nn.Module):
    def __init__(self, dim):
        super(Diagonal, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(1, self.dim))
        self.bias = nn.Parameter(torch.zeros(1, self.dim))

    def forward(self, x):
        return x * self.weight + self.bias

class EEGClassifier(nn.Module):
    def __init__(self, dim, n_nodes = 61):
        super(EEGClassifier, self).__init__()
     
    def forward(self, x):
        x = x.mean(1)
        return torch.sigmoid(x)

class EEGCondPrior(nn.Module):
    def __init__(self, dim, n_nodes = 61):
        super(EEGCondPrior, self).__init__()
        self.diag_loc_true = nn.Parameter(0.1*torch.randn(1, n_nodes, dim))
        self.diag_loc_false = nn.Parameter(0.1*torch.randn(1, n_nodes, dim))
        self.diag_scale_true = nn.Parameter(1 + 0.1*torch.randn(1, n_nodes, dim))
        self.diag_scale_false = nn.Parameter(1 + 0.1*torch.randn(1, n_nodes, dim))

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1,61,1)
        loc = x * self.diag_loc_true + (1 - x) * self.diag_loc_false
        scale = x * self.diag_scale_true + (1 - x) * self.diag_scale_false   
        return (6*x - 3) + torch.tanh(loc), torch.clamp(F.softplus(scale), min=1e-3)
    
class EEGEncoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            View((-1,1,61,100)),
            nn.Conv2d(1, 128, kernel_size=(11,9), stride=(1,2), padding=(11//2,9//2)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(11,5), stride=(1,2), padding=(11//2,5//2)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(11,5), stride=(1,2), padding=(11//2,5//2)),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=(11,5), stride=(1,2), padding=(11//2,5//2)),
            nn.ReLU(),
            View((-1,61,64*7)),
        )
                
        self.locs = nn.Linear(64*7, z_dim)
        self.scales = nn.Linear(64*7, z_dim)

    def forward(self, x):
        x = self.encoder(x)   
        return 3*torch.tanh(self.locs(x)), torch.clamp(F.softplus(self.scales(x)), min=1e-3)
        
class EEGDecoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        
        self.locs = nn.Sequential(
            nn.Linear(z_dim, 32*6),
            nn.ReLU(),
            View((-1,32,61,6)),
            nn.Conv2d(32,32,(11,5),(1,1),(11//2,5)),
            nn.Upsample(scale_factor=(1,2),  mode="bilinear", align_corners=False),
            nn.ReLU(),
            nn.Conv2d(32,32,(11,5),(1,1),(11//2,3)),
            nn.Upsample(scale_factor=(1,2),  mode="bilinear", align_corners=False),
            nn.ReLU(),
            nn.Conv2d(32,16,(11,5),(1,1),(11//2,2)),
            nn.Upsample(scale_factor=(1,2),  mode="bilinear", align_corners=False),
            nn.ReLU(),
            nn.Conv2d(16,1,(11,5),(1,1),(11//2,2)),
            View((-1,61,104)),
        )
        
        self.fc_scales = nn.Sequential(
            nn.Linear(z_dim, 32),
        )
        self.m_scales = nn.Parameter(torch.randn(32, 1))

    def forward(self, x):
        locs = self.locs(x)[:,:,2:-2]
        scales = torch.matmul(self.fc_scales(x), self.m_scales)
        return locs, torch.clamp(F.softplus(scales), min=1e-3)
