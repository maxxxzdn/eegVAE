import math
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F

import utils

#μ(X), σ(X) for q(Z|X)
class Encoder_x(nn.Module):
    def __init__(self, x_dim, n_nodes, latent_dim_z):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, stride=6, padding=0),
            utils.Flatten())
        
        self.n_nodes = n_nodes
        self.hidden_dim = 16*(x_dim//(2*2*2*6))
        
        self.mean_z = nn.Linear(self.hidden_dim, latent_dim_z)
        self.logvar_z = nn.Linear(self.hidden_dim, latent_dim_z)      
        
    def encode(self, x):
        x_encoded = self.encoder(x.unsqueeze(1))
        return x_encoded.view(-1, self.n_nodes, self.hidden_dim)
       
    def forward(self, x):
        x_encoded = self.encode(x)
        return self.mean_z(x_encoded), (0.5*self.logvar_z(x_encoded)).exp()

#μ(A), σ(A) for q(w|A)
class Encoder_a(nn.Module):
    def __init__(self, n_nodes, latent_dim_w):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,3,2,1),
            nn.ReLU(),
            nn.Conv2d(32,64,5,2,2),
            nn.ReLU(),
            nn.Conv2d(64,32,5,2,2),
            nn.ReLU(),
            nn.Conv2d(32,16,5,2,2),
            nn.ReLU(),
            utils.Flatten()
        )
        
        self.hidden_dim = 16*(math.ceil((n_nodes)/16))**2
        
        self.mean_w = nn.Linear(self.hidden_dim, latent_dim_w)
        self.logvar_w = nn.Linear(self.hidden_dim, latent_dim_w)
        
    def encode(self, a):
        return self.encoder(a.unsqueeze(1))
    
    def forward(self, z):
        z_encoded = self.encode(z)
        return self.mean_w(z_encoded), (0.5*self.logvar_w(z_encoded)).exp()

#μ(Z,A), σ for p(X|Z,A)   
class Decoder_za(nn.Module):
    def __init__(self, x_dim, n_nodes, latent_dim_z):
        super().__init__()
        self.latent_dim_z = latent_dim_z
        self.n_nodes = n_nodes
        self.x_dim = x_dim
        
        self.hidden_dim = math.ceil(x_dim/16)
        self.padding = ((16*self.hidden_dim - 4) - self.x_dim)//2
        
        self.gnn0 = gnn.GraphConv(latent_dim_z, 32)
        self.gnn1 = gnn.GraphConv(32, 32)
        self.gnn2 = gnn.GraphConv(32, self.hidden_dim)
        self.cnn =  nn.Sequential(  
            nn.Conv1d(1,32,3,1,1),
            nn.Upsample(scale_factor=2,  mode="linear", align_corners=False),
            nn.ReLU(),
            nn.Conv1d(32,32,5,1,2),
            nn.Upsample(scale_factor=2,  mode="linear", align_corners=False),
            nn.ReLU(),
            nn.Conv1d(32,32,5,1,2),
            nn.Upsample(scale_factor=2,  mode="linear", align_corners=False),
            nn.ReLU(),
            nn.Conv1d(32,32,5,1,2),
            nn.Upsample(scale_factor=2,  mode="linear", align_corners=False),
            nn.ReLU(),
            nn.Conv1d(32,1,5,1),
        )
        
        self.logvar_x = utils.define_param()

    def forward(self, z, a):
        edge_index, edge_weight = utils.dense_to_sparse(torch.where(a>0,a,torch.zeros(1).to(a.device)))
        h1 = F.relu(self.gnn0(z.view(-1,self.latent_dim_z), edge_index, edge_weight))
        h2 = F.relu(self.gnn1(h1, edge_index, edge_weight))
        h3 = F.relu(self.gnn2(h2, edge_index, edge_weight))       
        h = h3.unsqueeze(1)
        x = self.cnn(h).squeeze(1)[:,self.padding:-self.padding]
        return x.view(-1, self.n_nodes, self.x_dim), (0.5*self.logvar_x).exp()
    
#μ(w), σ for p(A|w)      
class Decoder_w(nn.Module):
    def __init__(self, latent_dim_w, n_nodes, mode):
        super().__init__()
        self.logvar_a = utils.define_param()
        self.hidden_dim = math.ceil(n_nodes/16)
        self.output_dim = (((math.ceil(n_nodes/16)*2+1)*2-1)*2-1)*2-1        
        self.padding = (self.output_dim - n_nodes)//2, (self.output_dim - n_nodes)//2 + n_nodes
        self.mode = mode
        if mode == 'conv':
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim_w, 16*self.hidden_dim**2),
                nn.ReLU(),
                utils.UnFlatten('w', self.hidden_dim),
                nn.ConvTranspose2d(16,32,5,2,1),
                nn.ReLU(),
                nn.ConvTranspose2d(32,64,5,2,2),
                nn.ReLU(),
                nn.ConvTranspose2d(64,32,5,2,2),
                nn.ReLU(),
                nn.ConvTranspose2d(32,1,3,2,1),
                utils.ToSymmetric(False)
            )
        elif mode == 'mlp':
            self.decoder = nn.Sequential(
            nn.Linear(latent_dim_w, 400),
            nn.ReLU(),
            nn.Linear(400, (n_nodes*n_nodes - n_nodes)//2),
            utils.ToSymmetric(True)) 
        elif mode == 'linear':
            self.decoder = nn.Sequential(
            nn.Linear(latent_dim_w, (n_nodes*n_nodes - n_nodes)//2),
            utils.ToSymmetric(True)) 

    def forward(self, w):
        if self.mode == 'conv':
            loc, scale = self.decoder(w).squeeze(1)[:, self.padding[0]:self.padding[1], self.padding[0]:self.padding[1]] , (0.5*self.logvar_a).exp()
        else:
            loc, scale = self.decoder(w).squeeze(1), (0.5*self.logvar_a).exp()
        return loc, scale
    
#μ(y), σ(y) for p(w_c|y)
class CondPrior_w(nn.Module):
    def __init__(self, laten_dim_w, mode):
        super(CondPrior_w, self).__init__()
        self.mode = mode
        if mode == 'nn':
            self.diag_loc_true = nn.Parameter(torch.zeros(1, laten_dim_w))
            self.diag_loc_false = nn.Parameter(torch.zeros(1, laten_dim_w))
            self.diag_scale_true = nn.Parameter(torch.ones(1, laten_dim_w))
            self.diag_scale_false = nn.Parameter(torch.ones(1, laten_dim_w))

    def forward(self, x):
        if self.mode == 'lookup':
            loc = 4*x - 2
            scale = torch.ones_like(loc)
        elif self.mode == 'nn':
            loc = x * self.diag_loc_true + (1 - x) * self.diag_loc_false
            scale = x * self.diag_scale_true + (1 - x) * self.diag_scale_false
            scale = torch.clamp(F.softplus(scale), min=1e-3)
        return loc, scale
    
#μ(y), σ(y) for p(Z_c|y)
class CondPrior_Z(nn.Module):
    def __init__(self, n_nodes, laten_dim_z, mode):
        super(CondPrior_Z, self).__init__()
        self.mode = mode
        self.ones = torch.ones(1, n_nodes, laten_dim_z)
        if mode == 'nn':
            self.diag_loc_true = nn.Parameter(torch.zeros(1, n_nodes, laten_dim_z))
            self.diag_loc_false = nn.Parameter(torch.zeros(1, n_nodes, laten_dim_z))
            self.diag_scale_true = nn.Parameter(torch.ones(1, n_nodes, laten_dim_z))
            self.diag_scale_false = nn.Parameter(torch.ones(1, n_nodes, laten_dim_z))

    def forward(self, x):
        if self.mode == 'lookup':
            x = x.unsqueeze(1)*self.ones.to(x.device)
            loc = 4*x - 2
            scale = torch.ones_like(loc)
        elif self.mode == 'nn':
            loc = x * self.diag_loc_true + (1 - x) * self.diag_loc_false
            scale = x * self.diag_scale_true + (1 - x) * self.diag_scale_false
            scale = torch.clamp(F.softplus(scale), min=1e-3)
        return loc, scale
    
#λ(Z_c,w_c) for q(y|Z_c, w_c)
class Classifier(nn.Module):
    def __init__(self, n_nodes, dim_y):
        super(Classifier, self).__init__()
        self.aggr = nn.Linear(n_nodes, 1)
        self.weight_w = nn.Parameter(torch.randn(1, dim_y))
        self.diag1 = utils.Diagonal(dim_y)
        self.diag2 = utils.Diagonal(dim_y)

    def forward(self, Z_c, w_c):
        zc_aggr = F.relu(self.aggr(Z_c.permute(0,1,3,2))).squeeze(-1)
        zc_wc = F.relu(zc_aggr + self.weight_w*w_c)
        return torch.sigmoid(self.diag2(F.relu(self.diag1(zc_wc)))) 