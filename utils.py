import torch
import random
import torch.nn as nn
import copy
import numpy as np
from torch_geometric.utils import to_dense_adj, add_self_loops, dense_to_sparse
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import Data
from optimizer import *

def KLD(p, q):
    if p.family == q.family == 'Normal':
        return lambda mu, logvar: 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
    elif p.family == q.family == 'Laplace':
        return lambda mu, logb: -(-logb + math.log(p.b) - 1 + mu.abs()/p.b + 1/p.b*(logb-mu.abs()/logb.exp()).exp()).sum()

class Normal():
    def __init__(self, mu=None, var=None):
        self.mu = mu
        self.var = var
        self.family = 'Normal'
        
class Laplace():
    def __init__(self, mu=None, b=None):
        self.mu = mu
        self.b = b
        self.family = 'Laplace'

class CorrData():
    def __init__(self, parameters, active_nodes):
        self.n_nodes = parameters['n_nodes']
        self.noise_mu = parameters['noise_mu']
        self.len_sig = parameters['len_sig']
        self.phase_mu = parameters['phase_mu']
        if active_nodes is not None:
            self.active_nodes = random.sample(active_nodes, parameters['n_active_nodes'])
        else:
            self.active_nodes = random.sample(list(range(self.n_nodes)), parameters['n_active_nodes'])
        self.x = self.create_data()
        self.adj = self.get_adj(self.x)
        self.edge_index = dense_to_sparse(self.adj)[0]
        self.y = 0
   
    def add_sin(self, x):
        phase = self.phase_mu*np.random.randn(1)
        o = np.arange(-3*np.pi, 3*np.pi, 6*np.pi/self.len_sig)
        return np.sin(1*o + phase)
    
    def create_data(self):
        noise = self.noise_mu*np.random.randn(self.n_nodes, self.len_sig)
        sins = np.zeros_like(noise)
        ind = self.active_nodes
        sins[ind] = np.apply_along_axis(self.add_sin, -1, sins[ind]) 
        return noise + sins
    
    def get_adj(self, x):
        adj = (np.abs(np.corrcoef(x)) > 0.5)*1.0
        return torch.tensor(adj).long()
    
class CorrDataset():
    def __init__(self, parameters, active_nodes, n_graphs):
        self.parameters = parameters
        self.active_nodes = active_nodes
        self.n_graphs = n_graphs
        self.dataset = self.make_dataset()
        
    def make_dataset(self):
        dataset = []
        for _ in range(0, self.n_graphs//2):
            data = CorrData(self.parameters, self.active_nodes)
            data.y = 0
            dataset.append(self.data_to_Data(data))
        for _ in range(self.n_graphs//2, self.n_graphs):
            data = CorrData(self.parameters, None)
            data.y = 1
            dataset.append(self.data_to_Data(data))
        random.shuffle(dataset)
        return dataset
    
    def data_to_Data(self, data):
        x = torch.tensor(data.x).float()
        adj = data.adj
        edge_index = data.edge_index
        y = torch.tensor(data.y).long()
        return Data(x = x, adj = adj, edge_index = edge_index, y = y)

def _sparseness(x):
    """Hoyer's measure of sparsity for a vector"""
    sqrt_n = np.sqrt(len(x))
    return (sqrt_n - np.linalg.norm(x, 1) / np.sqrt(squared_norm(x))) / (sqrt_n - 1)

def batch_sparseness(x, eps=1e-6):
    x = x/(x.std(0) + eps) #normalization, see https://arxiv.org/pdf/1812.02833.pdf
    return np.apply_along_axis(_sparseness, -1, x)

def squared_norm(x):
    """Squared Euclidean or Frobenius norm of x.

    Returns the Euclidean norm when x is a vector, the Frobenius norm when x
    is a matrix (2-d array). Faster than norm(x) ** 2.
    """
    x = np.ravel(x)
    return np.dot(x, x)
    
class Tracker():
    def __init__(self):
        self.loss = 0
        self.BCE = 0
        self.KLD = 0
        self.l1_loss = 0
    def update(self, loss, BCE, KLD, l1_loss):
        self.loss += loss.item()
        self.BCE += BCE.item()
        self.KLD += KLD.item()
        if l1_loss.__class__ is torch.Tensor:
            self.l1_loss += l1_loss.item()
        else: 
            self.l1_loss += l1_loss
    def get_mean(self, N):
        self.loss /= N
        self.BCE /= N
        self.KLD /= N
        self.l1_loss /= N
    def get_losses(self):
        return [self.loss, self.BCE, self.KLD, self.l1_loss]
    
class Log():
    def __init__(self):
        self.loss = []
        self.BCE = []
        self.KLD = []
        self.l1_loss = []
    def append(self, losses):
        loss, BCE, KLD, l1_loss = losses
        self.loss.append(loss)
        self.BCE.append(BCE)
        self.KLD.append(KLD)
        self.l1_loss.append(l1_loss)

class Logger():
    def __init__(self):
        self.train = Log()
        self.test = Log()
        self.best_epoch = 0
        self.best_test_loss = 0
        self.sparseness = 0
    def append(self, train_losses, test_losses):
        self.train.append(train_losses)
        self.test.append(test_losses)

def clones(module, N, shared=False):
    "Produce N identical layers (modules)."
    if shared:
        return nn.ModuleList(N*[module])
    else:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def create_mask(n_rows):
    """
    A mask to mask off the elements on diagonal for a square matrix with n_rows.
    Args:
        n_rows (int): number of rows/columns in the mask.
    Returns:
        square matrix with 0 on diagonal and 1 everywhere of size [1,n_rows,n_rows].
    """
    mask = torch.eye(n_rows).unsqueeze(0)
    zero_indices = mask == 0
    non_zero_indices = mask == 1
    mask[non_zero_indices] = 0
    mask[zero_indices] = 1
    return mask

def get_max_n_nodes(dataset):
    """
    Find a graph with maximum number of nodes in the dataset.'
    Args: 
        dataset (list): list of torch_geometric.data.Data objects.
    Returns:
        maximum number of nodes presented in the dataset.
    """
    return max([data.x.shape[0] for data in dataset])
                   
def update_dataset(dataset, max_n_nodes, dim_u):
    """
    Adds dense adjacency matrix for each graph in the dataset.
    Args:
        dataset (list): list of torch_geometric.data.Data objects.
        max_n_nodes (int): add max_n_nodes - |v| shallow isolated nodes to each graph.
    Returns:
        updated dataset with additional dense adjacency matrix.
    """
    dataset_ = []
    mask = create_mask(max_n_nodes)
    for data in dataset:
        # Add missing shallow isolated nodes 
        x = torch.cat([data.x, torch.zeros(max_n_nodes - data.x.shape[0],data.x.shape[1])],0)
        # Add self loops to make isolated nodes recognisable for DataLoader
        edge_index = add_self_loops(data.edge_index, num_nodes = max_n_nodes)[0]
        try:
            edge_index_ = add_self_loops(data.edge_index_, num_nodes = max_n_nodes)[0]
        except: 
            edge_index_ = None
        try:
            edge_attr = torch.cat([data.edge_attr, torch.zeros(edge_index.shape[1] - data.edge_index.shape[1], data.edge_attr.shape[1])],0)
        except:
            edge_attr = None
        y = data.y
        adj = to_dense_adj(edge_index)*mask
        try:
            adj_ = to_dense_adj(edge_index_)*mask
        except:
            adj_ = None
        if dim_u is not None:
            u = torch.zeros([len(y), dim_u])
        else:
            u = None
        dataset_.append(Data(x = x, edge_index = edge_index, edge_attr = edge_attr, u = u, y = y, adj = adj, adj_ = adj_))
    return dataset_

def cluster_centre(coordinates):
    """
    Finds centre of mass for a given distribution.
    Args:
        coordinates (tensor): coordinates of sampled points.
    Returns:
        centre of mass of given sampled points.
    """
    return coordinates.mean(0)

def get_cluster(model, dataset, y):
    """
    Maps instances of a class to latent space.
    Args: 
        model (nn.Module): VAE model.
        dataset (list): list of torch_geometric.data.Data objects.
        y (int): class label.
    Returns:
        tensor with coordinates in latent space for each instance of the class.
    """
    dataset_y = [data for data in dataset if data.y == y]
    loader_y = DataLoader(dataset_y, batch_size=10**10, shuffle=True)
    data_y = next(iter(loader_y))
    if model.type == 'mp':
        a,b = model.encode(data.x, data.edge_index, data.edge_attr, data.u, data.batch)
    elif model.type == 'adj':
        a,b = model.encode(data_y.adj.view(-1, model.n_nodes*model.n_nodes))
    else: 
        a,b = model.encode(data_y.x, data_y.edge_index, data_y.batch)
    z = model.reparameterize(a,b)
    return z

def digitize_z(z, n_bins = 10):
    """
    Digitizes a continuous variable z.
    Args:
        z (torch.tensor): continuous variable sampled.
        n_bins (int): number of bins to discretize z across.
    Returns:
        discretized z.
    """
    bins = np.linspace(z.min().item(), z.max().item(), 10)
    digitized = np.digitize(z, bins)
    return digitized

def trian_elements(tensor):
    """
    Returns upper triangular matrix elemets of a tensor of shape [B,N,N].
    Args:
        tensor (torch.tensor): tensor of shape [B,N,N].
    Returns:
        upper trianglar matrix elements of shape [B,(N^2-N)/2].
    """
    n_rows = tensor.shape[1]
    return tensor.permute(2,1,0)[torch.tril_indices(n_rows,n_rows, offset = -1).unbind()].T

def tensor_from_trian(trian):
    """
    Reconstructs a symmetric tensor of shape [B,N,N] from its triangular matrix.
    Args:
        trian (torch.tensor): tensor of shape [B,(N^2-N)/2].
    Returns:
        tensor of shape [B,N,N].
    """
    n_elements = trian.shape[1]
    # calculate number of rows in the original tensor
    n_rows = int((1 + np.sqrt(1+8*n_elements))//2) 
    tensor = torch.zeros(trian.shape[0], n_rows, n_rows).to(trian.device)
    # fill upper triangular elements: 
    tensor.permute(2,1,0)[torch.tril_indices(n_rows,n_rows, offset = -1).unbind()] = trian.T 
    # mirror upper triangular elements below diagonal
    tensor = (tensor.permute(2,1,0) + tensor.permute(1,2,0)).permute(2,0,1)
    return tensor

def add_edge(edge_index, i, j):
    """
    Adds undirected edge (i->j, j->i) to sparse adjacency matrix.
    Args:
        edge_index (torch.tensor): sparse adjacency matrix.
        i,j (int): number of nodes connected by the edge.
    Returns:
        updated sparse adjacency matrix.
    """
    to = torch.tensor([i,j])
    out = torch.tensor([j,i])
    edge_index = torch.cat([edge_index, to.unsqueeze(1)], 1)
    edge_index = torch.cat([edge_index, out.unsqueeze(1)], 1)
    return edge_index

def get_copy(data):
    """
    Returns a copy of Data object containing x, y, edge_attr, edge_index.
    Args:
        data (torch_geometric.data.Data): graph as Data object. 
    Returns:
        copy of data with edge_index_ attribute as copied edge_index.
    """
    edge_index = data.edge_index.detach().clone()
    edge_attr = data.edge_attr.detach().clone()
    x = data.x.detach().clone()
    y = data.y.detach().clone()
    return Data(x = x, y = y, edge_index = edge_index, edge_attr = edge_attr, edge_index_ = edge_index)
    
def add_edge_noise(dataset, n, m):
    """
    Adds undirected edge (i->j, j->i) to sparse adjacency matrix.
    Args:
        edge_index (torch.tensor): sparse adjacency matrix.
        i,j (int): number of nodes connected by the edge.
    Returns:
        updated sparse adjacency matrix.
    """
    dataset_ = copy.deepcopy(dataset)
    for _ in range(n):
        data = random.choice(dataset)
        data_ = get_copy(data)
        for i in range(m):
            k = random.randint(0, data_.edge_index.max())
            j = random.randint(0, data_.edge_index.max())
            if not in_adjacency(data_.edge_index, [k, j]):
                data_.edge_index = add_edge(data_.edge_index, k, j)
        dataset_.append(data_)
    return dataset_

def in_adjacency(edge_index, edge):
    """
    Checks if the edge is in the given adjacency matrix.
    Args:
        edge_index (torch.tensor): sparse adjacency matrix.
        edge (list [int,int]): edge of form [node1, node2].
    Returns:
        True if edge is in edge_index, False else.
    """
    to = edge[0]
    out = edge[1]
    intsec = np.intersect1d(np.where(edge_index[0] == to), np.where(edge_index[1] == out))
    return len(intsec)