import torch
import torch.nn as nn
import copy
import numpy as np
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from optimizer import *

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
        edge_attr = torch.cat([data.edge_attr, torch.zeros(edge_index.shape[1] - data.edge_index.shape[1], data.edge_attr.shape[1])],0)
        y = data.y
        u = torch.zeros([len(y), dim_u])
        adj = to_dense_adj(edge_index)*mask
        dataset_.append(Data(x = x, edge_index = edge_index, edge_attr = edge_attr, u = u, y = y, adj = adj))
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