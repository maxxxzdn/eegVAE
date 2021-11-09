class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1,32,12)

#Encoder
class Encoder_x(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=3, padding=3),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=7, stride=3, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=3, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=7, stride=3, padding=3),
            nn.ReLU(),
            Flatten())       
        
    def forward(self, data):
        x_encoded = self.encoder(data.x.unsqueeze(1))
        return x_encoded

class Encoder_z(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(2*latent_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 1)
        
    def forward(self, z):
        z = concatenate_combinations(z.view(16,61,-1))
        z_encoded = self.fc2(F.relu(self.fc1(z)))
        z_encoded = torch.sigmoid(z_encoded.view(16,61,61))
        return z_encoded
    
class Decoder_z(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.decoder = nn.Sequential(  
            nn.Linear(latent_dim, 32*12),
            nn.ReLU(),
            UnFlatten(),
            nn.Conv1d(32,16,3,1,1),
            nn.Upsample(scale_factor=3,  mode="linear", align_corners=False),
            nn.ReLU(),
            nn.Conv1d(16,8,5,1,1),
            nn.Upsample(scale_factor=3,  mode="linear", align_corners=False),
            nn.ReLU(),
            nn.Conv1d(8,4,5,1,1),
            nn.Upsample(scale_factor=3,  mode="linear", align_corners=False),
            nn.ReLU(),
            nn.Conv1d(4,1,5,1,2),)
    def forward(self, z):
        x_hat = self.decoder(z).squeeze() #, edge_index)
        return x_hat
    
class Decoder_w(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            Matrix(61),
            nn.ReLU(),
            Matrix(61))
        self.mask = torch.ones(61, 61).tril(-1).cuda()
        
    def forward(self, w):
        a = self.mask*torch.sigmoid(self.decoder(w))
        return a + a.permute(0,2,1)
    
class Matrix(nn.Module):
    def __init__(self, n_rows):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(n_rows,n_rows))
        self.b = torch.nn.Parameter(torch.randn(n_rows,n_rows))
        self.W.requires_grad = True
        self.b.requires_grad = True
    def forward(self, input):
        return self.W*input + self.b

#VAE    
class VAE1(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_x = Encoder_x(latent_dim)
        self.decoder_z = Decoder_z(latent_dim)
        self.encoder_z = Encoder_z(latent_dim)
        self.decoder_w = Decoder_w()
        
        self.loc_z = nn.Linear(64, latent_dim)
        self.logscale_z = nn.Linear(64, latent_dim)
        
        self.loc_w = Matrix(61)
        self.logscale_w = Matrix(61)
        
        self.logscale_x = nn.Parameter(torch.Tensor([0.0]))
        self.logscale_x.requires_grad = True
        
        self.logscale_a = nn.Parameter(torch.Tensor([0.0]))
        self.logscale_a.requires_grad = True
                
    def forward(self, data):
        #x_encoded
        x_encoded = self.encoder_x(data)
        #μ(x)
        loc_z = self.loc_z(x_encoded)
        #σ(x)
        scale_z = (0.5*self.logscale_z(x_encoded)).exp()
        #q(z|x)
        qz_x = torch.distributions.Normal(loc_z, scale_z)
        #z ~ q(z|x)                         
        z = qz_x.rsample()      
        #z_encoded
        z_encoded = self.encoder_z(z)
        #μ(z)
        loc_w = self.loc_w(z_encoded)
        #σ(z)
        scale_w = (0.5*self.logscale_w(z_encoded)).exp()
        #q(w|z)
        qw_z = torch.distributions.Normal(loc_w, scale_w)
        #w ~ q(w|z)
        w = qw_z.rsample()
        #x_decoded 
        x_hat = self.decoder_z(z)
        #a_decoded 
        a_hat = self.decoder_w(w)
        return x_hat, a_hat, z, w, loc_z, scale_z, loc_w, scale_w
        
    def loss_function(self, x, a, x_hat, a_hat, z, w, loc_z, scale_z, loc_w, scale_w):
        x = x.view(16,61,-1)
        a = a.view(16,61,-1)
        x_hat = x_hat.view(16,61,-1)
        a_hat = a_hat.view(16,61,-1)
        z = z.view(16,61,-1)
        w = w.view(16,61,-1)        
        loc_z = loc_z.view(16,61,-1)
        loc_w = loc_w.view(16,61,-1)  
        scale_z = scale_z.view(16,61,-1)
        scale_w = scale_w.view(16,61,-1)  
        

        recon_loss_x = self.likelihood_x(x, x_hat)
        recon_loss_a = self.likelihood_a(a, a_hat)
        kl_z = self.kl_divergence(z, loc_z, scale_z)
        kl_w = self.kl_divergence(w, loc_w, scale_w)
        
        elbo = (kl_z + kl_w - recon_loss_x - recon_loss_a)
        elbo = elbo.mean()
        
        return elbo, recon_loss_x.mean(), recon_loss_a.mean(), kl_w.mean()
    
    def likelihood_x(self, x, x_hat):
        scale = torch.exp(0.5*self.logscale_x)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)
        # measure prob of seeing image under p(x|z)
        log_px_z = dist.log_prob(x)
        return log_px_z.sum(dim=(1,2))
        
    def likelihood_a(self, a, a_hat):
        scale = torch.exp(0.5*self.logscale_a)
        mean = a_hat
        dist = torch.distributions.Normal(mean, scale)
        # measure prob of seeing image under p(a|w)
        log_pa_w = dist.log_prob(a)
        return log_pa_w.sum(dim=(1,2))
    
    def kl_divergence(self, z, mu, std):

        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qzx - log_pz)
        kl = kl.sum(dim=(1,2))
        return kl
