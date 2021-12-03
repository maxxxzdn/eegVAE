import torch
from torch import nn
import torch.nn.functional as F
from utils import Flatten, Aggregator, MMD_DIM, Cov
from sparse import Sparse
import torch.distributions as dist
import math

class ToSymmetric(nn.Module):
    def __init__(self, from_triang):
        super().__init__()
        self.from_triang = from_triang
    def forward(self, input):
        if not self.from_triang:
            input = input.tril(-1)
            return input + input.permute(0,1,3,2)
        else:
            output = tensor_from_trian(input.unsqueeze(0))
            return output.squeeze()
        
class UnFlatten(nn.Module):
    def __init__(self, var, size=None):
        super().__init__()
        self.var = var
        self.size = size
    def forward(self, input):
        if self.var == 'z':
            return input.view(input.shape[0]*input.shape[1], -1, self.size)
        elif self.var == 'w':
            return input.view(-1, 64, 1, 1)

class Encoder_x(nn.Module):
    def __init__(self, n_nodes, latent_dim_z, drop_rate = 0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=7, stride=3, padding=3),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=7, stride=3, padding=3),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=7, stride=3, padding=3),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5, stride=2),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            Flatten())
        
        self.mean_z = nn.Linear(48, latent_dim_z)
        self.logvar_z = nn.Linear(48, latent_dim_z)      
        
    def encode(self, x):
        x_encoded = self.encoder(x.unsqueeze(1))
        return x_encoded.view(-1, self.n_nodes, 48)
       
    def forward(self, x):
        x_encoded = self.encode(x)
        return self.mean_z(x_encoded), (0.5*self.logvar_z(x_encoded)).exp()

    
class Encoder_z(nn.Module):
    def __init__(self, latent_dim_z, hidden_dim, latent_dim_w, n_nodes, drop_rate):
        super().__init__()
        self.encoder = nn.Sequential(
            Aggregator('none'),
            nn.Linear(n_nodes*latent_dim_z, hidden_dim), 
            nn.Dropout(drop_rate),
            nn.ReLU())
        
        self.mean_w = nn.Linear(hidden_dim, latent_dim_w)
        self.logvar_w = nn.Linear(hidden_dim, latent_dim_w)
        
    def encode(self, z):
        return self.encoder(z)
    
    def forward(self, z):
        z_encoded = self.encode(z)
        return self.mean_w(z_encoded), (0.5*self.logvar_w(z_encoded)).exp()
            
            
class Decoder_z(nn.Module):
    def __init__(self, latent_dim, n_nodes, drop_rate):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes
        self.decoder = nn.Sequential(  
            nn.Linear(latent_dim, 32*12),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            UnFlatten('z', 12),
            nn.Conv1d(32,16,3,1,2),
            nn.Dropout(drop_rate),
            nn.Upsample(scale_factor=2,  mode="linear", align_corners=False),
            nn.ReLU(),
            nn.Conv1d(16,8,5,1,2),
            nn.Dropout(drop_rate),
            nn.Upsample(scale_factor=2,  mode="linear", align_corners=False),
            nn.ReLU(),
            nn.Conv1d(8,8,5,1,2),
            nn.Dropout(drop_rate),
            nn.Upsample(scale_factor=3,  mode="linear", align_corners=False),
            nn.ReLU(),
            nn.Conv1d(8,4,5,1,2),
            nn.Dropout(drop_rate),
            nn.Upsample(scale_factor=3,  mode="linear", align_corners=False),
            nn.ReLU(),
            nn.Conv1d(4,1,5,1),)
    def forward(self, z):
        x_hat = self.decoder(z).squeeze()
        return x_hat.view(-1, self.n_nodes, 500)
                 
    
class Decoder_w(nn.Module):
    def __init__(self, latent_dim, n_nodes, drop_rate, mode):
        super().__init__()
        if mode == 'conv':
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.Dropout(drop_rate),
                nn.ReLU(),
                UnFlatten('w', 64),
                nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
                nn.Dropout(drop_rate),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2),
                nn.Dropout(drop_rate),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2),
                nn.Dropout(drop_rate),
                nn.ReLU(),
                nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2))
        elif mode == 'mlp':
            self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            nn.Linear(400, (n_nodes*n_nodes - n_nodes)//2),
            ToSymmetric(True)) 
        elif mode == 'linear':
            self.decoder = nn.Sequential(
            nn.Linear(latent_dim, (n_nodes*n_nodes - n_nodes)//2),
            ToSymmetric(True)) 
        
    def forward(self, w):
        return self.decoder(w).squeeze()
    
def define_param():
    param = nn.Parameter(torch.Tensor([0.0]))
    param.requires_grad = True
    return param

class CondPrior(nn.Module):
    def __init__(self, y_dim, w_dim):
        super(CondPrior, self).__init__()
        self.fc1 = nn.Linear(y_dim, w_dim//2)
        self.loc = nn.Linear(w_dim//2, w_dim)
        self.logvar = nn.Linear(w_dim//2, w_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.loc(x), (0.5*self.logvar(x)).exp()

class Classifier(nn.Module):
    def __init__(self, dim_w, dim_y):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim_w, dim_w*2),
            nn.ReLU(),
            nn.Linear(dim_w*2, dim_y))

    def forward(self, x):
        return torch.softmax(self.classifier(x), -1)
    
class ccVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_nodes = args.n_nodes
        self.n_classes = args.n_classes
        self.scale_z = args.scale_z
        
        self.encoder_x = Encoder_x(args.n_nodes, args.latent_dim_z, args.drop_rate)
        self.decoder_z = Decoder_z(args.latent_dim_z, args.n_nodes, args.drop_rate)
        self.encoder_z = Encoder_z(args.latent_dim_z, args.hidden_dim_enc_z, args.latent_dim_w, args.n_nodes, args.drop_rate)
        self.decoder_w = Decoder_w(args.latent_dim_w, args.n_nodes, args.drop_rate, args.decoder_w_mode)
        self.classifier = Classifier(args.latent_dim_w, args.n_classes)
        self.cond_prior = CondPrior(args.n_classes, args.latent_dim_w)
                
        self.logvar_x = define_param()
        self.logvar_a = define_param()
        
        self.y_prior_params = torch.ones(1, self.n_classes) / 2
        
    def reconstruct_a(self, x):
        z = dist.Normal(*self.encoder_x(x)).rsample() 
        w = dist.Normal(*self.encoder_z(z)).rsample()
        return self.decoder_w(w)
        
    def reconstruct_x(self, x):
        return self.decoder_z(dist.Normal(*self.encoder_x(x)).rsample())
    
    def reconstruct_data(self, data):
        return self.reconstruct_x(data.x), self.reconstruct_a(data.x)
        
    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
                                        
    def forward(self, data):
        #μ(x), σ(x)
        mean_z, stddev_z = self.encoder_x(data.x)
        #q(z|x)
        qz_x = dist.Normal(mean_z, stddev_z)
        #z ~ q(z|x)                         
        z = qz_x.rsample()  
        #μ(z), σ(z)
        mean_w, stddev_w = self.encoder_z(z)
        #q(w|z)
        qw_z = dist.Normal(mean_w, stddev_w)
        #w ~ q(w|z)
        w = qw_z.rsample()
        #μ(w)
        mean_x = self.decoder_z(z)
        #p(x|z)
        px_z = dist.Normal(mean_x, (0.5*self.logvar_x).exp())
        #μ(w)
        mean_a = self.decoder_w(w)
        #p(a|w)
        pa_w = dist.Normal(mean_a, (0.5*self.logvar_a).exp())
        #λ(w)
        probs_y = self.classifier(w)
        #q(y|w)
        qy_wc = dist.ContinuousBernoulli(probs_y)
        #μ(y)
        mean_w_prior, stddev_w_prior = self.cond_prior(data.y)
        #p(w|y)
        pw_y = dist.Normal(mean_w_prior, stddev_w_prior)

        return px_z, qz_x, qy_wc, pa_w, qw_z, pw_y, mean_a, z, w
        
    def loss_function(self, args, use_w):
        x, a, y, px_z, qz_x, qy_wc, pa_w, qw_z, pw_y, z, w = args
        bs = x.shape[0]
        pz = dist.Normal(0, self.scale_z)   
        
        log_px_z = px_z.log_prob(x).sum(-1).sum(-1)
        log_pa_w = pa_w.log_prob(a).sum(-1).sum(-1)
        log_qy_wc = qy_wc.log_prob(y).sum(-1)
        
        kl_z = self.calculate_kl(z, qz_x, pz).sum(-1).sum(-1)
        kl_w = self.calculate_kl(w, qw_z, pw_y).sum(-1)
        
        log_qy_x = self.classifier_loss(x, y)
        log_py = dist.ContinuousBernoulli(self.y_prior_params.to(x.device).expand(bs, -1)).log_prob(y).sum(dim=-1)
                
        w = torch.clamp((log_qy_wc - log_qy_x).exp().detach(), 0., 5.)  if use_w else 1.
                
        elbo = w*(log_px_z - kl_z + log_pa_w - kl_w + log_qy_wc) + log_qy_x + log_py     
        
        return -elbo.mean(), log_px_z.mean(), log_pa_w.mean(), log_qy_x.mean(), kl_z.mean(), kl_w.mean()
    
    def calculate_kl(self, latent, posterior, prior):
        log_posterior = posterior.log_prob(latent)
        log_prior = prior.log_prob(latent)
        return log_posterior - log_prior
        
    
    def classifier_loss(self, x, y, k_z=100, k_w=100):
        """
        Computes the classifier loss.
        """
        x = x.view(-1,500)
        z = dist.Normal(*self.encoder_x(x)).rsample([k_z]) 
        w = dist.Normal(*self.encoder_z(z)).rsample([k_w])
        probs_y = self.classifier(w)
        d = dist.ContinuousBernoulli(probs_y)
        y = y.expand(k_w, k_z, -1, -1,).contiguous()
        lqy_w = d.log_prob(y).sum(-1)
        lqy_x = torch.logsumexp(lqy_w, dim=[0,1]) - math.log(k_z) - math.log(k_w)
        return lqy_x
