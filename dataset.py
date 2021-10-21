import copy
import random
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, to_dense_adj, add_self_loops
from utils import create_mask, add_edge, in_adjacency, get_copy


class CorrData():
    def __init__(self, parameters, active_nodes):
        """A graph with nodes carrying either noise or sinusoid + noise
        Args:
            parameters (dict): dict with the following parameters:
                n_nodes (int): number of nodes in the graph.
                noise_std (float): std of noise.
                len_sig (int): duration of the signal.
                phase_std (float): std of phase of sinusoid.
            active_nodes (list): list of nodes that can carry sinusoids.
        """
        self.n_nodes = parameters['n_nodes']
        self.noise_std = parameters['noise_std']
        self.len_sig = parameters['len_sig']
        self.phase_std = parameters['phase_std']
        if active_nodes is not None:
            self.active_nodes = random.sample(active_nodes, parameters['n_active_nodes'])
        else:
            self.active_nodes = random.sample(list(range(self.n_nodes)), parameters['n_active_nodes'])
        self.x = self.create_data()
        self.adj = self.get_adj(self.x)
        self.edge_index = dense_to_sparse(self.adj)[0]
        self.y = 0
    def add_sin(self, x):
        """
        Create a sinusoid of the same length as x.
        Args:
            x (np.array): signal matrix of shape [n_nodes, len_signal].
        Returns:
            a sinusoid of shape [1, len_signal].
        """
        phase = self.phase_std*np.random.randn(1)
        time = np.arange(-3*np.pi, 3*np.pi, 6*np.pi/self.len_sig)
        return np.sin(1*time + phase)   
    def create_data(self):
        """
        Construct signal matrix of shape [n_nodes, len_signal].
        Signal is noisy sinusoid if node in active_nodes else noise.
        """
        noise = self.noise_std*np.random.randn(self.n_nodes, self.len_sig)
        sins = np.zeros_like(noise)
        ind = self.active_nodes
        sins[ind] = np.apply_along_axis(self.add_sin, -1, sins[ind]) 
        return noise + sins
    @staticmethod
    def get_adj(x):
        """
        Construct adjacency matrix from a raw signal.
        Args:
            x (np.array): signal matrix of shape [n_nodes, len_signal].
        Returns:
            Adjacency matrix with 1 if corr coefficient > 0.5 else 0.
        """
        adj = (np.abs(np.corrcoef(x)) > 0.5)*1.0
        return torch.tensor(adj).long()
    
class CorrDataset():
    def __init__(self, parameters, active_nodes, n_graphs):
        """
        Args:
            parameters (dict): dictionary with parameters for CorrData.
            active_nodes (list): list of nodes that carry signal.
            n_graphs (int): number of graphs in the dataset.
        """
        self.parameters = parameters
        self.active_nodes = active_nodes
        self.n_graphs = n_graphs
        self.dataset = self.make_dataset()        
    def make_dataset(self):
        """
        Creates a dataset.
        """
        dataset = []
        for _ in range(0, self.n_graphs//2):
            data = CorrData(self.parameters, self.active_nodes)
            data.y = 0
            dataset.append(self.to_data(data))
        for _ in range(self.n_graphs//2, self.n_graphs):
            data = CorrData(self.parameters, None)
            data.y = 1
            dataset.append(self.to_data(data))
        random.shuffle(dataset)
        return dataset   
    @staticmethod
    def to_data(data):
        """
        Turns CorrData instance into torch_geometric.data.Data
        Args: 
            data (CorrData): instance of CorrData class
        Returns:
            torch_geometric.data.Data instance
        """
        x = torch.tensor(data.x).float()
        adj = data.adj
        edge_index = data.edge_index
        y = torch.tensor(data.y).long()
        return Data(x = x, adj = adj, edge_index = edge_index, y = y)
    
def get_max_n_nodes(dataset):
    """
    Find a graph with maximum number of nodes in the dataset.
    Args: 
        dataset (list): list of torch_geometric.data.Data objects.
    Returns:
        maximum number of nodes presented in the dataset.
    """
    return max([data.x.shape[0] for data in dataset])
                   
def update_dataset(dataset, max_n_nodes):
    """
    Adds dense adjacency matrix for each graph in the dataset.
    Args:
        dataset (list): list of torch_geometric.data.Data objects.
        max_n_nodes (int): add max_n_nodes - |v| shallow isolated nodes to each graph.
    Returns:
        updated dataset with additional dense adjacency matrix.
    """
    datase_upd = []
    mask = create_mask(max_n_nodes)
    for data in dataset:
        # Add missing shallow isolated nodes 
        x = torch.cat([data.x, torch.zeros(max_n_nodes - data.x.shape[0],data.x.shape[1])],0)
        # Add self loops to make isolated nodes recognisable for DataLoader
        edge_index = add_self_loops(data.edge_index, num_nodes = max_n_nodes)[0]
        try:
            ei_orig = add_self_loops(data.ei_orig, num_nodes = max_n_nodes)[0]
            adj_orig = to_dense_adj(ei_orig)*mask
        except: 
            adj_orig = None
        adj = to_dense_adj(edge_index)*mask
        datase_upd.append(Data(x = x, edge_index = edge_index, y = data.y, adj = adj, adj_orig = adj_orig))
    return datase_upd

def add_edge_noise(dataset, n_graphs, n_edges):
    """
    Adds graphs with noisy edges to a dataset.
    Args:
        dataset (list): list of torch_geometric.data.Data objects.
        n_graphs (int): number of graphs to add.
        n_edges (int): number of noisy edges to add.
    Returns:
        updated dataset with some graphs containing edge noise.
    """
    dataset_copy = copy.deepcopy(dataset)
    for _ in range(n_graphs):
        data = random.choice(dataset)
        data_copy = get_copy(data)
        for _ in range(n_edges):
            k = random.randint(0, data_copy.edge_index.max())
            j = random.randint(0, data_copy.edge_index.max())
            if not in_adjacency(data_copy.edge_index, [k, j]):
                data_copy.edge_index = add_edge(data_copy.edge_index, k, j)
        dataset_copy.append(data_copy)
    return dataset_copy
