import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

    
def performance_plot(log):
    plt.plot(log["train_loss"], label = 'train loss')
    plt.plot(log["test_loss"], label = 'test loss')
    plt.vlines(x = np.argmin(log["test_loss"]), ymin = 0, ymax = np.max(log["test_loss"]), linestyle='dashed', label = 'best performance')
    plt.legend()
    plt.show()
    
def show_graphs(adjacency_matrix, ax = None):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, ax = ax)
    
def visualize_adj_graph(p_adj, threshold = 0.5):   
    adj = (p_adj > threshold) * 1.
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = [15,5])
    img = ax1.imshow(p_adj)
    ax1.set_title('p(A)')
    ax2.imshow(adj)
    ax2.set_title('A reconstructed')
    show_graphs(adj, ax = ax3)
    ax3.set_title('Graph reconstructed')    
    plt.colorbar(img, ax=ax1, fraction=0.046, pad=0.04)
    plt.show()
    
def visualize_recon_adj(recovered, adj, threshold = 0.5):
    recovered_ = (recovered > threshold) * 1.
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = [15,5])
    ax1.imshow(adj)
    ax1.set_title('A true')
    ax2.imshow(recovered_)
    ax2.set_title('A reconstructed')
    img = ax3.imshow(recovered)
    ax3.set_title('p(A)')    
    plt.colorbar(img, ax=ax3,fraction=0.046, pad=0.04)
    plt.show()
    
def visualize_recon_graph(recovered, adj, threshold = 0.5):
    recovered_ = (recovered > threshold) * 1.
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize = [10,6])
    show_graphs(adj, ax = ax1)
    ax1.set_title('Graph true')
    show_graphs(recovered_, ax = ax2)
    ax2.set_title('Graph reconstructed')

def visualize_z_space(z, y = None, z_centre = None):
    plt.scatter(z[:,0], z[:,1], c = y)
    plt.title('Latent space')
    plt.xlabel('$z_{1}$')
    plt.ylabel('$z_{2}$')
    if z_centre is not None:
        x_centre, y_centre = z_centre
        plt.scatter(x = x_centre, y = y_centre, c = 'red', marker = 'x', s = 200, label = 'cluster centre')
    plt.legend()
    
def vis_digitized(continuous, digitized): 
    f, (ax1, ax2) = plt.subplots(1, 2, figsize = [10,5])
    ax1.scatter(y = continuous, x = range(len(continuous)))
    ax1.set_xlabel('# graph')
    ax1.set_ylabel('$ z_1 $')
    ax1.set_title('Distribution of continuous $z_1$')
    ax2.scatter(y = digitized, x = range(len(digitized)))
    ax2.set_xlabel('# graph')
    ax2.set_ylabel('discretized $ z_1 $')
    ax2.set_title('Distribution of discrete $z_1$')