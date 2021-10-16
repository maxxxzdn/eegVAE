import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.nn import MetaLayer
from utils import *
from meta import *
import math
        
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
        bs = max(batch) + 1
        hidden1 = F.relu(F.dropout(self.gc1(x, edge_index), self.dropout))
        hidden2 = F.relu(F.dropout(self.gc2(hidden1, edge_index), self.dropout))
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

class Decoder(nn.Module):
    def __init__(self, n_nodes, latent_dim, hidden_dim, dropout):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, (n_nodes*n_nodes - n_nodes)//2)
        
    def __call__(self, z):
        h = F.relu(F.dropout(self.fc1(z), self.dropout))
        return torch.sigmoid(self.fc2(h))
        
class VAE(nn.Module):
    def __init__(self, encoder, decoder, prior, posterior, KLD, beta, gamma):
        super(VAE, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.posterior = posterior
        self.KLD = KLD(prior, posterior)
        self.latent_dim = encoder.latent_dim
        self.n_nodes = encoder.n_nodes
        self.type = encoder.type
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x, **kwargs):
        mu, logvar = self.encoder(x, kwargs)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z
    
    def loss_function(self, preds, labels, parameter1, parameter2, z):
        BCE = F.binary_cross_entropy(preds, labels, reduction='sum')
        KLD = self.KLD(parameter1, parameter2)  
        l1_loss = F.l1_loss(z, zeros_like(z), reduction = 'sum')
        return BCE + self.beta*KLD + self.gamma*l1_loss, BCE, KLD, l1_loss
       
def train(model, optimizer, loader, gamma = 0.0, mode = True):
    model.train(mode = mode)
    train_tracker = Tracker()
    sparseness = []
    for data in loader:
        optimizer.zero_grad()
        if model.type == 'mp':
            recovered, p1, p2, z = model(data.x, data.edge_index, data.edge_attr, data.u, data.batch)
        elif model.type == 'adj':
            recovered, p1, p2, z = model(x = data.adj.view(-1, model.n_nodes**2))
        else:
            recovered, p1, p2, z = model(x = data.x, edge_index = data.edge_index, batch = data.batch)  
            
        adj = data.adj.reshape(-1, model.n_nodes, model.n_nodes)
        adj = trian_elements(adj)
        
        loss, BCE, KLD, l1_loss = model.loss_function(preds=recovered, labels=adj,
                             parameter1=p1, parameter2=p2, z=z)
        sparseness += batch_sparseness(z.detach().numpy()).tolist()
        loss.backward()
        train_tracker.update(loss, BCE, KLD, l1_loss)
        if mode:
            optimizer.step()
            
    train_tracker.get_mean(len(loader.dataset))
    return train_tracker.get_losses(), np.mean(sparseness)

def fit(model, optimizer, train_loader, test_loader, epochs, save_file, gamma):
    best_loss = np.inf
    log = Logger()
    for epoch in tqdm(range(epochs), desc="Training for {} epochs".format(epochs)):
        train_losses, sparseness = train(model, optimizer, train_loader, gamma, True)
        test_losses, sparseness = train(model, optimizer, test_loader, gamma, False)
        if test_losses[0] < best_loss:
            best_loss = test_losses[0]
            log.best_test_loss = best_loss
            log.best_epoch = epoch
            log.sparseness = sparseness
            torch.save(model.state_dict(), save_file)            
        log.append(train_losses, test_losses)
    print("Optimization Finished!")
    print("Model type: " + model.type)
    print('Best epoch: {}'.format(log.best_epoch), ', Best test set loss: {:.4f}'.format(best_loss))
    print('Sparseness: {:.3f}'.format(log.sparseness))
    return log

"""
class convVAE(nn.Module):
    def __init__(self, n_nodes, input_feat_dim, hidden_dim1, hidden_dim2, beta, dropout):
        super(convVAE, self).__init__()
        self.n_nodes = n_nodes
        self.latent_dim = hidden_dim2
        self.type = 'conv'
        self.beta = beta
        self.dropout = dropout
        self.gc1 = gnn.GraphConv(input_feat_dim, hidden_dim1)
        self.gc2 = gnn.GraphConv(hidden_dim1, hidden_dim2)
        self.gc3 = gnn.GraphConv(hidden_dim1, hidden_dim2)
        self.fc1 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, (n_nodes*n_nodes - n_nodes)//2)

    def encode(self, x, edge_index, batch):
        hidden1 = F.relu(F.dropout(self.gc1(x, edge_index), self.dropout))
        mu = gnn.global_mean_pool(self.gc2(hidden1, edge_index), batch)
        logvar = gnn.global_mean_pool(self.gc3(hidden1, edge_index), batch)
        return mu.squeeze(), logvar.squeeze()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decode(self, z):
        h = F.relu(F.dropout(self.fc1(z), self.dropout))
        return torch.sigmoid(self.fc2(h))
        
    def forward(self, x, edge_index, batch):
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
    
class chebVAE(nn.Module):
    def __init__(self, n_nodes, input_feat_dim, hidden_dim1, hidden_dim2, beta, K = 7):
        super(chebVAE, self).__init__()
        self.n_nodes = n_nodes
        self.latent_dim = hidden_dim2
        self.type = 'cheb'
        self.beta = beta
        self.gc1 = gnn.ChebConv(input_feat_dim, hidden_dim1, K)
        self.gc2 = gnn.ChebConv(hidden_dim1, hidden_dim2, K)
        self.gc3 = gnn.ChebConv(hidden_dim1, hidden_dim2, K)
        self.fc1 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, (n_nodes*n_nodes - n_nodes)//2)

    def encode(self, x, edge_index, batch):
        hidden1 = F.relu(self.gc1(x, edge_index))
        mu = gnn.global_mean_pool(self.gc2(hidden1, edge_index), batch)
        logvar = gnn.global_mean_pool(self.gc3(hidden1, edge_index), batch)
        return mu.squeeze(), logvar.squeeze()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decode(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))
        
    def forward(self, x, edge_index, batch):
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
    
class attVAE(nn.Module):
    def __init__(self, n_nodes, input_feat_dim, hidden_dim1, hidden_dim2, beta):
        super(attVAE, self).__init__()
        self.n_nodes = n_nodes
        self.latent_dim = hidden_dim2
        self.type = 'att'
        self.beta = beta
        self.gc1 = gnn.GATConv(input_feat_dim, hidden_dim1)
        self.gc2 = gnn.GATConv(hidden_dim1, hidden_dim2)
        self.gc3 = gnn.GATConv(hidden_dim1, hidden_dim2)
        self.fc1 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, (n_nodes*n_nodes - n_nodes)//2)

    def encode(self, x, edge_index, batch):
        hidden1 = F.relu(self.gc1(x, edge_index))
        mu = gnn.global_mean_pool(self.gc2(hidden1, edge_index), batch)
        logvar = gnn.global_mean_pool(self.gc3(hidden1, edge_index), batch)
        return mu.squeeze(), logvar.squeeze()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decode(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))
        
    def forward(self, x, edge_index, batch):
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
    
class mpVAE(nn.Module):
    def __init__(self, n_nodes, dims, beta):
        super(mpVAE, self).__init__()
        self.n_nodes = n_nodes
        self.latent_dim = dims["u_f"]
        self.type = 'mp'     
        self.beta = beta
        self.gc1 = MetaLayer(EdgeModel(dims), NodeModel(dims), GlobalModel(dims))
        self.gc2 = MetaLayer(EdgeModel(dims), NodeModel(dims), GlobalModel(dims))
        self.gc3 = MetaLayer(EdgeModel(dims), NodeModel(dims), GlobalModel(dims))      
        self.fc1 = nn.Linear(dims["u_f"], 100)
        self.fc2 = nn.Linear(100, (n_nodes*n_nodes - n_nodes)//2)

    def encode(self, x, edge_index, edge_attr, u, batch):
        x, edge_attr, u = self.gc1(x, edge_index, edge_attr, u, batch)
        x = F.relu(x)
        edge_attr = F.relu(edge_attr)
        u = F.relu(u)
        _, _, mu = self.gc2(x, edge_index, edge_attr, u, batch)
        _, _, logvar = self.gc3(x, edge_index, edge_attr, u, batch)
        return mu.squeeze(), logvar.squeeze()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decode(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))
        
    def forward(self, x, edge_index, edge_attr, u, batch):
        mu, logvar = self.encode(x, edge_index, edge_attr, u, batch)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z



class GCNModelVAE(nn.Module):
    def __init__(self, n_nodes, n_layers, input_feat_dim, hidden_dim1, hidden_dim2):
        super(GCNModelVAE, self).__init__()
        self.n_nodes = n_nodes
        self.latent_dim = hidden_dim2
        self.gc1 = gnn.GraphConv(input_feat_dim, hidden_dim1)
        self.gc2 = clones(gnn.GraphConv(hidden_dim1, hidden_dim1), n_layers)
        self.fc_mu = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_var = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc1 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, n_nodes*n_nodes)

    def encode(self, x, edge_index, batch):
        h = F.relu(self.gc1(x, edge_index))
        for gc in self.gc2:
            h = F.relu(gc(h, edge_index))
        h = gnn.global_mean_pool(h, batch)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decode(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))
        
    def forward(self, x, edge_index, batch):
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
"""