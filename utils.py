import torch
import numpy as np
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from optimizer import *

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
                   
def update_dataset(dataset, max_n_nodes = None):
    """
    Adds dense adjacency matrix for each graph in the dataset.
    Args:
        dataset (list): list of torch_geometric.data.Data objects.
        max_n_nodes (int): if not None, add max_n_nodes - |v| shallow isolated nodes to each graph.
    Returns:
        updated dataset with additional dense adjacency matrix.
    """
    dataset_ = []
    for data in dataset:
        # Add missing shallow isolated nodes 
        x = torch.cat([data.x, torch.zeros(max_n_nodes - data.x.shape[0],7)],0)
        # Add self loops to make isolated nodes recognisable for DataLoader
        edge_index = add_self_loops(data.edge_index, num_nodes = max_n_nodes)[0]
        y = data.y
        adj = to_dense_adj(edge_index)
        dataset_.append(Data(x = x, edge_index = edge_index, y = y, adj = adj))
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