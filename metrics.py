import numpy as np
import torch

def calculate_sparseness(loader, model):   
    """Hoyer's measure of sparsity of latent representation for a dataset"""
    sparseness = []
    model.train(mode = False)
    with torch.no_grad():
        for data in loader:
            latent = model(data)[-1].detach().numpy()
            sparseness += batch_sparseness(latent).tolist()
    return np.mean(sparseness)

def calculate_cov(loader, model):
    """Covariance of latent representation for a dataset"""
    cov = 0
    model.train(mode = False)
    with torch.no_grad():
        for data in loader:
            latent = model(data)[-1].detach().numpy()
            cov_batch = np.abs(latent.T)
            cov += cov_batch[np.tril_indices(model.latent_dim, k = -1)].mean()
    return cov/(len(loader))
                          
def _sparseness(x):
    """Hoyer's measure of sparsity for a vector"""
    sqrt_n = np.sqrt(len(x))
    return (sqrt_n - np.linalg.norm(x, 1) / np.sqrt(squared_norm(x))) / (sqrt_n - 1)

def batch_sparseness(x, eps=1e-6):
    """Hoyer's measure of sparsity for a batch"""
    x = x/(x.std(0) + eps) #normalization, see https://arxiv.org/pdf/1812.02833.pdf
    return np.apply_along_axis(_sparseness, -1, x)

def squared_norm(x):
    """Squared Euclidean or Frobenius norm of x.

    Returns the Euclidean norm when x is a vector, the Frobenius norm when x
    is a matrix (2-d array). Faster than norm(x) ** 2.
    """
    x = np.ravel(x)
    return np.dot(x, x)