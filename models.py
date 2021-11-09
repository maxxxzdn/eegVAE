import torch
from torch import nn, zeros_like, sigmoid
import torch.nn.functional as F
import torch_geometric.nn as gnn

      
class VAE(nn.Module):
    def __init__(self, encoder, decoder, prior, beta, gamma, rec_loss):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.latent_dim = encoder.latent_dim
        self.n_nodes = encoder.n_nodes
        self.type = encoder.type
        self.rec_loss = rec_loss
        self.logscale = nn.Parameter(torch.Tensor([0.0]))
    def get_scale_tril(self, logscale):
        a = torch.zeros((logscale.shape[0], self.latent_dim, self.latent_dim))
        b = torch.diag_embed((logscale[:,:self.latent_dim]).exp())
        c = torch.diag_embed(logscale[:,self.latent_dim:], -1)
        return a + b + c
    def get_scale_mat(self, logscale):
        l = self.get_scale_tril(logscale)
        return torch.bmm(l, l.permute(0,2,1))
    def reparameterize(self, loc, logscale):
        if type(self.prior).__name__ != 'MultivariateNormal':
            q = self.prior.__class__(loc, (0.5*logscale).exp())
            return q.rsample()
        else:   
            eps = torch.randn(loc.shape)
            return loc + torch.bmm(self.get_scale_tril(logscale), eps.unsqueeze(-1)).squeeze() if self.training else loc
    def forward(self, data):
        loc, logscale = self.encoder(data)
        latent = self.reparameterize(loc, logscale)
        return self.decoder(latent), loc, logscale, latent
    def loss_function(self, preds, labels, loc, logscale, latent, beta):
        recon_loss = self.gaussian_likelihood(preds, labels)
        kl = self.KLD(latent, loc, logscale)
        elbo = (beta*kl - recon_loss)
        return elbo.mean(), recon_loss.mean(), kl.mean(), 0
    
    def KLD(self, z, loc, logscale):
        if type(self.prior).__name__ == 'MultivariateNormal':
            posterior_cov = self.get_scale_mat(logscale)
            posterior_loc = loc
            posterior = self.prior.__class__(posterior_loc, posterior_cov)
        else: 
            posterior_loc = loc
            posterior_scale = (0.5*logscale).exp()
            posterior = self.prior.__class__(posterior_loc, posterior_scale)
            
        #log_qzx = posterior.log_prob(z)
        #log_pz = self.prior.log_prob(z)

        # kl
        #kl = (log_qzx - log_pz).sum(-1)
        kl = torch.distributions.kl_divergence(posterior, self.prior).sum(-1)
        return kl
    """
    def gaussian_likelihood(self, x_hat, x):
        scale = torch.exp(self.logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(-1)
    """
    def gaussian_likelihood(self, x_hat, x):
        lambda_cb = x_hat
        dist = torch.distributions.ContinuousBernoulli(lambda_cb)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x.float())
        return log_pxz.sum(-1)
                                            
class Decoder(nn.Module):
    def __init__(self, n_nodes, latent_dim, hidden_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, (n_nodes*n_nodes - n_nodes)//2)       
    def forward(self, latent):
        hidden = F.relu(F.dropout(self.fc1(latent), self.dropout))
        return torch.sigmoid(self.fc2(hidden))

class Encoder_conv_mlp(nn.Module):
    def __init__(self, n_nodes, input_dim, hidden_dim, latent_dim, dropout, multivariate):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.type = 'conv_mlp'
        self.gc1 = gnn.GraphConv(input_dim, hidden_dim)
        self.gc2 = gnn.GraphConv(hidden_dim, hidden_dim)
        self.loc = nn.Linear(n_nodes*hidden_dim, latent_dim)
        scale_dim = latent_dim if not multivariate else latent_dim + (latent_dim-1)
        self.logscale = nn.Linear(n_nodes*hidden_dim, scale_dim)
        
    def forward(self, data):
        batch_size = max(data.batch) + 1
        hidden1 = F.relu(F.dropout(self.gc1(data.x, data.edge_index), self.dropout))
        hidden2 = F.relu(F.dropout(self.gc2(hidden1, data.edge_index), self.dropout))
        loc = self.loc(hidden2.view(batch_size, -1))
        logscale = self.logscale(hidden2.view(batch_size, -1))
        return loc.squeeze(), logscale.squeeze()
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1) 
    
class Encoder_adj_conv(nn.Module):
    def __init__(self, n_nodes, input_dim, hidden_dim, latent_dim, dropout, multivariate):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.type = 'adj'
        self.fc = nn.Sequential(
                                nn.Conv2d(1, 16, kernel_size=5, stride=2),
                                nn.ReLU(),
                                nn.Conv2d(16, 32, kernel_size=5, stride=2),
                                nn.ReLU(),
                                nn.Conv2d(32, 8, kernel_size=5, stride=2),
                                nn.ReLU(),
                                Flatten(),
                                )
        in_dim = 200
        self.loc = nn.Linear(in_dim, latent_dim)
        scale_dim = latent_dim if not multivariate else latent_dim + (latent_dim-1)
        self.logscale = nn.Linear(in_dim, scale_dim)        
        
    def forward(self, data):
        adj = data.adj.view(-1, 1, self.n_nodes, self.n_nodes)
        hidden = F.relu(self.fc(adj))
        return self.loc(hidden), self.logscale(hidden)
    
           
class Encoder_adj(nn.Module):
    def __init__(self, n_nodes, input_dim, hidden_dim, latent_dim, dropout, multivariate):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.type = 'adj'
        self.fc = nn.Linear(n_nodes*n_nodes, hidden_dim)
        self.loc = nn.Linear(hidden_dim, latent_dim)
        scale_dim = latent_dim if not multivariate else latent_dim + (latent_dim-1)
        self.logscale = nn.Linear(hidden_dim, scale_dim)        
        
    def forward(self, data):
        adj = data.adj.view(-1, self.n_nodes**2)
        hidden = F.dropout(F.relu(self.fc(adj)), self.dropout)
        return self.loc(hidden), self.logscale(hidden)
    
class CVAE(VAE):
    def __init__(self, encoder, decoder, prior, beta, gamma, rec_loss):
        super().__init__(encoder, decoder, prior, beta, gamma, rec_loss)
    def forward(self, data):
        loc, logscale = self.encoder(data)
        latent = self.reparameterize(loc, logscale)
        return self.decoder(latent, data.y), loc, logscale, latent
    
class CEncoder_adj(Encoder_adj):
    def __init__(self, n_nodes, input_dim, hidden_dim, latent_dim, dropout, multivariate):
        super().__init__(n_nodes, input_dim, hidden_dim, latent_dim, dropout, multivariate) 
        self.fc = nn.Linear(n_nodes*n_nodes+2, hidden_dim)
    def forward(self, data):
        adj = data.adj.view(-1, self.n_nodes**2)
        x_y = torch.cat([adj, F.one_hot(data.y, 2)],1)
        hidden = F.dropout(F.relu(self.fc(x_y)), self.dropout)
        return self.loc(hidden), self.logscale(hidden)
        
class CDecoder(Decoder):
    def __init__(self, n_nodes, latent_dim, hidden_dim, dropout):
        super().__init__(n_nodes, latent_dim, hidden_dim, dropout)
        self.fc1 = nn.Linear(latent_dim+2, hidden_dim)
    def forward(self, latent, y):
        latent_y = torch.cat([latent, F.one_hot(y, 2)],1)
        hidden = F.relu(F.dropout(self.fc1(latent_y), self.dropout))
        return sigmoid(self.fc2(hidden))  