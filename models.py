from torch import nn, zeros_like, sigmoid
import torch.nn.functional as F
import torch_geometric.nn as gnn
    
class VAE(nn.Module):
    def __init__(self, encoder, decoder, prior, beta, gamma):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.latent_dim = encoder.latent_dim
        self.n_nodes = encoder.n_nodes
        self.type = encoder.type      
    def reparameterize(self, loc, logscale):
        scale = (0.5*logscale).exp()
        eps = self.prior.sample(loc.shape)
        return loc + eps*scale if self.training else loc
    def forward(self, data):
        loc, logscale = self.encoder(data)
        latent = self.reparameterize(loc, logscale)
        return self.decoder(latent), loc, logscale, latent
    def loss_function(self, preds, labels, loc, logscale, latent):
        BCE = F.binary_cross_entropy(preds, labels, reduction='sum')
        KLD = self.KLD(loc, logscale)
        l1_loss = F.l1_loss(latent, zeros_like(latent), reduction = 'sum') if self.gamma else 0
        return (BCE - self.beta*KLD) + self.gamma*l1_loss, BCE, KLD, l1_loss
    def KLD(self, loc, logscale):
        if type(self.prior).__name__ == 'Normal':
            prior_var = self.prior.variance
            prior_mu = self.prior.mean
            return -0.5*(prior_var.log() + 1/prior_var*(logscale.exp() + loc**2) - 1 - logscale).sum()
        elif type(self.prior).__name__ == 'Laplace':
            prior_b = self.prior.scale
            prior_mu = self.prior.loc            
            return -(logscale.exp()/prior_b*(-loc.abs()/logscale.exp()).exp() + loc.abs()/prior_b + prior_b.log() - logscale - 1).sum()
        
class Decoder(nn.Module):
    def __init__(self, n_nodes, latent_dim, hidden_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, (n_nodes*n_nodes - n_nodes)//2)       
    def forward(self, latent):
        hidden = F.relu(F.dropout(self.fc1(latent), self.dropout))
        return sigmoid(self.fc2(hidden))

class Encoder_conv_mlp(nn.Module):
    def __init__(self, n_nodes, input_dim, hidden_dim, latent_dim, dropout):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.type = 'conv_mlp'
        self.gc1 = gnn.GraphConv(input_dim, hidden_dim)
        self.gc2 = gnn.GraphConv(hidden_dim, hidden_dim)
        self.loc = nn.Linear(n_nodes*hidden_dim, latent_dim)
        self.logscale = nn.Linear(n_nodes*hidden_dim, latent_dim)
        
    def forward(self, data):
        batch_size = max(data.batch) + 1
        hidden1 = F.relu(F.dropout(self.gc1(data.x, data.edge_index), self.dropout))
        hidden2 = F.relu(F.dropout(self.gc2(hidden1, data.edge_index), self.dropout))
        loc = self.loc(hidden2.view(batch_size, -1))
        logscale = self.logscale(hidden2.view(batch_size, -1))
        return loc.squeeze(), logscale.squeeze()
    
class Encoder_adj(nn.Module):
    def __init__(self, n_nodes, input_dim, hidden_dim, latent_dim, dropout):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.type = 'adj'
        self.fc = nn.Linear(n_nodes*n_nodes, hidden_dim)
        self.loc = nn.Linear(hidden_dim, latent_dim)
        self.logscale = nn.Linear(hidden_dim, latent_dim)        
        
    def forward(self, data):
        adj = data.adj.view(-1, self.n_nodes**2)
        hidden = F.relu(self.fc(adj))
        return self.loc(hidden), self.logscale(hidden)