import torch
from torch import nn
import torch.nn.functional as F
from utils import Flatten, UnFlatten, Aggregator, ToSymmetric

        
class cEncoder_x(nn.Module):
    def __init__(self, n_nodes, n_classes, drop_rate = 0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_classes = n_classes
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
        
    def forward(self, x, y):
        x_encoded = self.encoder(x.unsqueeze(1))
        x_encoded = torch.cat([x_encoded, y], -1)
        return x_encoded.view(-1, self.n_nodes, 48+self.n_classes)

    
class cEncoder_z(nn.Module):
    def __init__(self, latent_dim, n_nodes, n_classes, drop_rate, aggr_z):
        super().__init__()
        self.latent_dim = latent_dim
        self.aggr = Aggregator(aggr_z)
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim+n_classes, 400) if aggr_z != 'none' else nn.Linear(n_nodes*latent_dim+n_classes, 400),
            nn.Dropout(drop_rate),
            nn.ReLU())
        
    def forward(self, z, y):
        z = torch.cat([self.aggr(z), y], -1)
        return self.fc1(z)
            
            
class cDecoder_z(nn.Module):
    def __init__(self, latent_dim, n_nodes, n_classes, drop_rate):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes
        self.decoder = nn.Sequential(  
            nn.Linear(latent_dim+n_classes, 32*12),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            UnFlatten('z'),
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
    def forward(self, z, y):
        z = torch.cat([z,y],-1)
        x_hat = self.decoder(z).squeeze()
        return x_hat.view(-1, self.n_nodes, 500)
                 
    
class cDecoder_w(nn.Module):
    def __init__(self, latent_dim, n_nodes, n_classes, drop_rate, mode):
        super().__init__()
        if mode == 'conv':
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim+n_classes, 64),
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
                nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2),
                ToSymmetric(False))
        elif mode == 'mlp':
            self.decoder = nn.Sequential(
            nn.Linear(latent_dim+n_classes,400),
            nn.Dropout(drop_rate),
            nn.ReLU(),
            nn.Linear(400, (n_nodes*n_nodes - n_nodes)//2),
            ToSymmetric(True)) 
        
    def forward(self, w, y):
        w = torch.cat([w,y],-1)
        return self.decoder(w).permute(1,0,2,3)
    

class cVAE(nn.Module):
    def __init__(self, args, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.m = args.m
        self.n_nodes = args.n_nodes
        self._likelihood_a = args.likelihood_a
        self.encoder_x = cEncoder_x(args.n_nodes, n_classes, args.drop_rate)
        self.decoder_z = cDecoder_z(args.latent_dim_z, args.n_nodes, n_classes, args.drop_rate)
        self.encoder_z = cEncoder_z(args.latent_dim_z, args.n_nodes, n_classes, args.drop_rate, args.aggr_z)
        self.decoder_w = cDecoder_w(args.latent_dim_w, args.n_nodes, n_classes, args.drop_rate, args.decoder_w_mode)
        
        self.loc_z = nn.Linear(48+n_classes, args.latent_dim_z)
        self.logscale_z = nn.Linear(48+n_classes, args.latent_dim_z)
        
        self.loc_w = nn.Linear(400, args.latent_dim_w)
        self.logscale_w = nn.Linear(400, args.latent_dim_w)
        
        self.logscale_x = nn.Parameter(torch.Tensor([0.0]))
        self.logscale_x.requires_grad = True
        
        self.logscale_a = nn.Parameter(torch.Tensor([0.0]))
        self.logscale_a.requires_grad = True
                
    def forward(self, data):
        #x_encoded
        y = F.one_hot(data.y, self.n_classes).repeat(self.n_nodes,1)
        x_encoded = self.encoder_x(data.x, y)
        #μ(x)
        loc_z = self.loc_z(x_encoded)
        #σ(x)
        scale_z = (0.5*self.logscale_z(x_encoded)).exp()
        #q(z|x)
        qz_x = torch.distributions.Normal(loc_z, scale_z)
        #z ~ q(z|x)                         
        z = qz_x.rsample()      
        #z_encoded
        y = F.one_hot(data.y, self.n_classes)
        z_encoded = self.encoder_z(z,y)
        #μ(z)
        loc_w = self.loc_w(z_encoded)
        #σ(z)
        scale_w = (0.5*self.logscale_w(z_encoded)).exp()
        #q(w|z)
        qw_z = torch.distributions.Normal(loc_w, scale_w)
        #w ~ q(w|z)
        w = qw_z.rsample((self.m,))
        #x_decoded 
        y = F.one_hot(data.y, self.n_classes).repeat(self.n_nodes,1).view(-1, self.n_nodes, self.n_classes)
        x_hat = self.decoder_z(z,y)
        #a_decoded 
        y = F.one_hot(data.y, self.n_classes).unsqueeze(0).repeat(self.m,1,1)
        a_hat = self.decoder_w(w,y)
        return x_hat, a_hat, z, w, loc_z, scale_z, loc_w, scale_w
        
    def loss_function(self, args, coefs):
        x, a, x_hat, a_hat, z, w, loc_z, scale_z, loc_w, scale_w = args
        alpha_x, alpha_a, beta_z, beta_w = coefs
        
        x = x.view(-1,self.n_nodes,500)
        a = a.unsqueeze(0)
        w = w.permute(1,0,2)
        loc_w = loc_w.unsqueeze(0).permute(1,0,2)
        scale_w = scale_w.unsqueeze(0).permute(1,0,2)
               
        recon_loss_x = alpha_x*self.likelihood_x(x, x_hat)
        recon_loss_a = alpha_a*self.likelihood_a(a, a_hat)
                            
        kl_z = beta_z*self.kl_divergence(z, loc_z, scale_z)
        kl_w = beta_w*self.kl_divergence(w, loc_w, scale_w)
        
        elbo = (kl_z - recon_loss_x) + (kl_w - recon_loss_a)
        elbo = elbo.mean()
        
        return elbo, recon_loss_x.mean(), recon_loss_a.mean(), kl_z.mean(), kl_w.mean()
    
    def likelihood_x(self, x, x_hat):
        scale = torch.exp(0.5*self.logscale_x)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)
        # measure prob of seeing image under p(x|z)
        log_px_z = dist.log_prob(x)
        return log_px_z.sum(-1).mean(-1)
        
    def likelihood_a(self, a, a_hat):
        if self._likelihood_a == 'Normal':
            scale = torch.exp(0.5*self.logscale_a)
            mean = a_hat
            dist = torch.distributions.Normal(mean, scale)
            # measure prob of seeing image under p(a|w)
            log_pa_w = dist.log_prob(a)
            return log_pa_w.sum(-1).sum(-1)
        elif self._likelihood_a == 'CB':
            lambda_cb = a_hat
            dist = torch.distributions.ContinuousBernoulli(lambda_cb.clip(0,1))
            # measure prob of seeing image under p(a|w)
            log_pa_w = dist.log_prob(a.float())
            return log_pa_w.sum(-1).sum(-1)
    
    def kl_divergence(self, z, mu, std):
        
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qzx - log_pz)
        kl = kl.sum(-1).mean(-1)
        return kl