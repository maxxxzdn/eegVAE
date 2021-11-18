import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

    
def performance_plot(log):
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize = [10,10])
    ax1.plot(log.train.elbo, label = 'train ELBO')
    ax1.plot(log.test.elbo, label = 'test ELBO')
    ax1.set_xlabel('epoch')
    ax1.vlines(x = np.argmin(log.test.elbo), ymin = 0, ymax = np.max(log.test.elbo), linestyle='dashed', label = 'best performance')
    ax1.legend()
    ax2.plot(log.train.recon_loss_x, label = 'train X reconstruction')
    ax2.plot(log.test.recon_loss_x, label = 'test X reconstruction')
    ax2.legend()
    ax2.set_xlabel('epoch')
    ax3.plot(log.train.recon_loss_a, label = 'train A reconstruction')
    ax3.plot(log.test.recon_loss_a, label = 'test A reconstruction')
    ax3.legend()
    ax3.set_xlabel('epoch')
    ax4.plot(log.train.kl_w, label = 'train $KL_W$')
    ax4.plot(log.test.kl_w, label = 'test $KL_W$')
    ax4.legend()
    ax4.set_xlabel('epoch')
    ax5.plot(log.train.kl_z, label = 'train $KL_Z$')
    ax5.plot(log.test.kl_z, label = 'test $KL_Z$')
    ax5.legend()
    ax5.set_xlabel('epoch')
    ax6.plot(log.train.mse_A, label = 'train $MSE(A,\hat A)$')
    ax6.plot(log.test.mse_A, label = 'test $MSE(A,\hat A)$')
    ax6.legend()
    ax6.set_xlabel('epoch')
    plt.show()
    
def show_graphs(adjacency_matrix, ax = None):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    if adjacency_matrix.shape[-1] == 61:
        pos = np.genfromtxt('../EEGraph/EEG_data/Easycap_Koordinaten_61CH.txt')[:,0:2]
        gr.remove_edges_from(nx.selfloop_edges(gr))
        nx.draw(gr, pos = pos, node_size=500, ax = ax)
    else:
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
    
def visualize_recon_adj(recovered, adj, threshold = 0.5, adj_clear=None):
    recovered_ = (recovered > threshold) * 1.
    if adj_clear is not None:
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = [10,10])
    else: 
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = [15,5])
    ax1.imshow(adj)
    ax1.set_title('A true')
    ax2.imshow(recovered_)
    ax2.set_title('A reconstructed')
    img = ax3.imshow(recovered)
    ax3.set_title('p(A)')    
    plt.colorbar(img, ax=ax3,fraction=0.046, pad=0.04)
    if adj_clear is not None:
        ax4.imshow(adj_clear)
        ax4.set_title('A clear')
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