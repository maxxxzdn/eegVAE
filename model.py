import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F

class GCNModelVAE(nn.Module):
    def __init__(self, n_nodes, input_feat_dim, hidden_dim1, hidden_dim2):
        super(GCNModelVAE, self).__init__()
        self.n_nodes = n_nodes
        self.gc1 = gnn.GraphConv(input_feat_dim, hidden_dim1)
        self.gc2 = gnn.GraphConv(hidden_dim1, hidden_dim2)
        self.gc3 = gnn.GraphConv(hidden_dim1, hidden_dim2)
        self.fc1 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, n_nodes*n_nodes)

    def encode(self, x, edge_index, batch):
        hidden1 = F.relu(self.gc1(x, edge_index))
        mu = gnn.global_mean_pool(self.gc2(hidden1, edge_index), batch)
        logvar = gnn.global_mean_pool(self.gc3(hidden1, edge_index), batch)
        return mu.squeeze(), logvar.squeeze()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))
        
    def forward(self, x, edge_index, batch):
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar