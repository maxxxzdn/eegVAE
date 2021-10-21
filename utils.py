import torch
import random
import torch.nn as nn
import copy
import numpy as np
from torch_geometric.data import Data, DataLoader
import itertools


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
        a,b = model.encoder(data.x, data.edge_index, data.edge_attr, data.u, data.batch)
    elif model.type == 'adj':
        a,b = model.encoder(data_y.adj.view(-1, model.n_nodes*model.n_nodes))
    else: 
        a,b = model.encoder(data_y.x, data_y.edge_index, data_y.batch)
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
    return Data(x = x, y = y, edge_index = edge_index, edge_attr = edge_attr, ei_orig = edge_index)
    

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