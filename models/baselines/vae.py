import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from ..fc_networks import (FCEncoder_v2, FCDecoder)


class VAE(nn.Module):
    """
    VAE + classification
    """
    def __init__(self, z_dim, num_classes,
                 use_cuda, mode, p_dropout, beta):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.z_classify = num_classes
        self.z_style = z_dim - num_classes
        self.num_classes = num_classes
        self.mode = mode
        self.beta = beta

        if mode == 'FC':
            self.encoder = FCEncoder_v2(z_dim, p_dropout)
            self.decoder = FCDecoder(z_dim, p_dropout)  
            self.classifier = nn.Sequential(
                nn.Linear(z_dim, z_dim//2),
                nn.ReLU(),
                nn.Linear(z_dim//2, self.num_classes),
                nn.Sigmoid(),
            )
        else:
            raise NotImplementedError("not implemented")
         
        if use_cuda:
            self.cuda()

    def elbo(self, x, y):
        """
        Computes ELBO + classification loss.
        """
        #μ, σ of q(z|x)
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample() 
        #KL(q(z|x) || p(z))
        kl = compute_kl(*post_params)
        #x ~ p(x|z)
        recon = self.decoder(z)
        #ELBO
        log_pxz = self.img_log_likelihood(recon, x)
        elbo = log_pxz - self.beta*kl
        #ELBO + classification
        y_pred = self.classifier(z)
        loss = -elbo + F.binary_cross_entropy(y_pred,y)
        return loss.mean()

    def reconstruct_img(self, x):
        """
        Computes μ of p(x|z).
        """
        return self.decoder(dist.Normal(*self.encoder(x)).sample())[0]

    def classifier_acc(self, x, y=None, k=1):
        """
        Computes accuracy of classification.
        """
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample()
        probs = self.classifier(z)
        preds = torch.round(probs)
        acc = (preds.eq(y)).float().mean(0)
        return acc
    
    def img_log_likelihood(self, recon, xs):
        """
        Computes log p(x|z).
        """        
        if self.mode == 'FC':
            return dist.Normal(*recon).log_prob(xs).sum(dim=(-1,-2))
        else:
            return dist.Normal(*recon).log_prob(xs).sum(dim=(-1))

def compute_kl(locs_q, scale_q, locs_p=None, scale_p=None):
    """
    Computes the KL(q||p)
    """
    if locs_p is None:
        locs_p = torch.zeros_like(locs_q)
    if scale_p is None:
        scale_p = torch.ones_like(scale_q)

    dist_q = dist.Normal(locs_q, scale_q)
    dist_p = dist.Normal(locs_p, scale_p)
    return dist.kl.kl_divergence(dist_q, dist_p).sum(dim=-1)