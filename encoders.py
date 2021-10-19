import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F

class Encoder_conv_mlp(nn.Module):
    def __init__(self, n_nodes, input_dim, hidden_dim, latent_dim, dropout):
        super(Encoder_conv_mlp, self).__init__()
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.type = 'conv_mlp'
        self.gc1 = gnn.GraphConv(input_dim, hidden_dim)
        self.gc2 = gnn.GraphConv(hidden_dim, hidden_dim)
        self.mu = nn.Linear(n_nodes*hidden_dim, latent_dim)
        self.logvar = nn.Linear(n_nodes*hidden_dim, latent_dim)
        
    def __call__(self, x, *args):
        batch = args[0]['batch']
        edge_index = args[0]['edge_index']
        edge_weight = args[0]['edge_weight']
        bs = max(batch) + 1
        hidden1 = F.relu(F.dropout(self.gc1(x, edge_index, edge_weight), self.dropout))
        hidden2 = F.relu(F.dropout(self.gc2(hidden1, edge_index, edge_weight), self.dropout))
        mu = self.mu(hidden2.view(bs, -1))
        logvar = self.logvar(hidden2.view(bs, -1))
        return mu.squeeze(), logvar.squeeze()
    
class Encoder_adj(nn.Module):
    def __init__(self, n_nodes, input_dim, hidden_dim, latent_dim, dropout):
        super(Encoder_adj, self).__init__()
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.type = 'adj'
        self.fc = nn.Linear(n_nodes*n_nodes, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)        
        
    def __call__(self, x, *args):
        h1 = F.relu(self.fc(x))
        return self.mu(h1), self.logvar(h1)