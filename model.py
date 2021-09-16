import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils import *

class GCNModelVAE(nn.Module):
    def __init__(self, n_nodes, input_feat_dim, hidden_dim1, hidden_dim2):
        super(GCNModelVAE, self).__init__()
        self.n_nodes = n_nodes
        self.latent_dim = hidden_dim2
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
    
def train(model, optimizer, loader, mask):
    model.train()
    train_loss = 0
    for data in loader:
        recovered, mu, logvar = model(data.x, data.edge_index, data.batch) 
        adj = data.adj.reshape(-1, model.n_nodes**2)*mask
        loss = loss_function(preds=recovered.reshape(-1, model.n_nodes**2)*mask, labels=adj,
                             mu=mu, logvar=logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return train_loss / len(loader.dataset)

def test(model, optimizer, loader, mask):
    model.train(mode=False)
    test_loss = 0
    for data in loader:
        recovered, mu, logvar = model(data.x, data.edge_index, data.batch) 
        adj = data.adj.reshape(-1, model.n_nodes**2)*mask
        loss = loss_function(preds=recovered.reshape(-1, model.n_nodes**2)*mask, labels=adj,
                             mu=mu, logvar=logvar)
        test_loss += loss.item()
        optimizer.step()

    return test_loss / len(loader.dataset)

def fit(model, optimizer, train_loader, test_loader, epochs, save_file):
    mask = create_mask(model.n_nodes).reshape(-1, model.n_nodes**2)
    best_loss = np.inf
    log = {'train_loss': [], 'test_loss': [], 'best_epoch': 0, 'best_test_loss': best_loss}
    for epoch in tqdm(range(epochs), desc="Training for {} epochs".format(epochs)):
        train_loss = train(model, optimizer, train_loader, mask)
        test_loss = test(model, optimizer, test_loader, mask)
        if test_loss < best_loss:
            best_loss = test_loss
            log['best_test_loss'] = best_loss
            log['best_epoch'] = epoch
            torch.save(model.state_dict(), save_file)            
        log['train_loss'].append(train_loss)
        log['test_loss'].append(test_loss)
    print("Optimization Finished!")
    print('Best epoch: {}'.format(log['best_epoch']), ', Best test set loss: {:.4f}'.format(best_loss))
    
    return log