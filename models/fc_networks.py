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

class FCClassifier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.sigmoid(x)

class FCCondPrior(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.diag_loc_true = nn.Parameter(torch.zeros(self.dim))
        self.diag_loc_false = nn.Parameter(torch.zeros(self.dim))
        self.diag_scale_true = nn.Parameter(torch.ones(self.dim))
        self.diag_scale_false = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):
        loc = x * self.diag_loc_true + (1 - x) * self.diag_loc_false
        scale = x * self.diag_scale_true + (1 - x) * self.diag_scale_false
        return (10*x - 5) + torch.tanh(loc), torch.clamp(F.softplus(scale), min=1e-3, max=1.)
    
class FCCondPrior_v2(nn.Module):
    def __init__(self, z_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_classes, z_dim),
            nn.ReLU(),
        )
        self.locs = nn.Linear(z_dim, z_dim)
        self.scales = nn.Linear(z_dim, z_dim)
    def forward(self, x):
        x = self.fc(x)
        return torch.clamp(self.locs(x), -5, 5), torch.clamp(F.softplus(self.scales(x)), min=1e-3, max=1.)
    
class FCEncoder(nn.Module):
    def __init__(self, z_dim, p_dropout):
        super().__init__()

        self.encoder = nn.Sequential(
            View((-1,1,64,64)),
            nn.Conv2d(1, 8, kernel_size=(5,5), stride=(2,2), padding=(2,2)),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(5,5), stride=(2,2), padding=(2,2)),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(5,5), stride=(2,2), padding=(2,2)),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(5,5), stride=(2,2), padding=(2,2)),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            View((-1,8*4*4)),
        )
                
        self.locs = nn.Linear(128, z_dim)
        self.scales = nn.Linear(128, z_dim)

    def forward(self, x):
        x = self.encoder(x)   
        return 10*torch.sigmoid(self.locs(x))-5, torch.clamp(F.softplus(self.scales(x)), min=1e-3, max=1.)
    
class FCEncoder_v2(nn.Module):
    def __init__(self, z_dim, p_dropout):
        super().__init__()

        self.encoder = nn.Sequential(
            View((-1,1,64,64)),
            nn.Conv2d(1, 16, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            View((-1,8*16)),
        )
                
        self.locs = nn.Linear(128, z_dim)
        self.scales = nn.Linear(128, z_dim)

    def forward(self, x):
        x = self.encoder(x)   
        return torch.clamp(self.locs(x), -5, 5), torch.clamp(F.softplus(self.scales(x)), min=1e-3, max=1.)
        
class FCDecoder(nn.Module):
    def __init__(self, z_dim, p_dropout):
        super().__init__()
        
        self.locs = nn.Sequential(
            nn.Linear(z_dim, 8*4*4),
            nn.ReLU(),
            View((-1,8,4,4)),
            nn.ConvTranspose2d(8,16,(5,5),(2,2),(3,3)),
            nn.ReLU(),
            nn.ConvTranspose2d(16,16,(5,5),(2,2),(2,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(16,8,(5,5),(2,2),(2,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(8,8,(5,5),(2,2),(2,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(8,1,(5,5),(2,2),(2,2)),
            View((-1,65,65))
        )
        
        self.fc_scales = nn.Sequential(
            nn.Linear(z_dim, 32),
        )
        self.m_scales = nn.Parameter(torch.randn(32, 1))

    def forward(self, x):
        locs = self.locs(x)[:,:64,:64]
        locs = locs.tril(-1)
        locs = locs + locs.transpose(-1,-2)
        scales = torch.matmul(self.fc_scales(x), self.m_scales).unsqueeze(-1)
        return locs, torch.clamp(F.softplus(scales), min=1e-3, max=1.)
