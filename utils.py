import torch
import numpy as np
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.data import Data
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

def performance_plot(log):
    plt.plot(log["train_loss"], label = 'train loss')
    plt.plot(log["test_loss"], label = 'test loss')
    plt.vlines(x = np.argmin(log["test_loss"]), ymin = 0, ymax = np.max(log["test_loss"]), linestyle='dashed', label = 'best performance')
    plt.legend()
    plt.show()