import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from .fc_networks import (FCEncoder, FCDecoder, FCClassifier, FCCondPrior)


class CCVAE(nn.Module):
    """
    CCVAE
    see https://arxiv.org/abs/2006.10102 for details
    """
    def __init__(self, z_dim, num_classes,
                 use_cuda, mode, p_dropout, beta):
        super(CCVAE, self).__init__()
        self.z_dim = z_dim
        self.z_classify = num_classes
        self.z_style = z_dim - num_classes
        self.num_classes = num_classes
        self.ones = torch.ones(1, self.z_style)
        self.zeros = torch.zeros(1, self.z_style)
        self.mode = mode
        self.beta = beta

        if mode == 'FC':
            self.encoder = FCEncoder(z_dim, p_dropout)
            self.decoder = FCDecoder(z_dim, p_dropout)  
            self.classifier = FCClassifier(self.num_classes)
            self.cond_prior = FCCondPrior(self.num_classes)        
        else:
            raise NotImplementedError("not implemented")

        if use_cuda:
            self.ones = self.ones.cuda()
            self.zeros = self.zeros.cuda()
            self.cuda()

    def elbo(self, x, y):
        """
        Computes ELBO.
        """
        bs = x.shape[0]
        # inference
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample()
        zc, zs = z.split([self.z_classify, self.z_style], -1)
        qyzc = dist.Bernoulli(probs=self.classifier(zc))
        log_qyzc = qyzc.log_prob(y).sum(dim=-1)

        # compute kl
        locs_p_zc, scales_p_zc = self.cond_prior(y)
        if self.mode == 'EEG':
            prior_params = (torch.cat([locs_p_zc, self.zeros.expand(bs, 61, -1)], dim=-1), 
                            torch.cat([scales_p_zc, self.ones.expand(bs, 61, -1)], dim=-1))
        else:
            prior_params = (torch.cat([locs_p_zc, self.zeros.expand(bs, -1)], dim=-1), 
                            torch.cat([scales_p_zc, self.ones.expand(bs, -1)], dim=-1))            
        kl = self.beta*compute_kl(*post_params, *prior_params)

        # compute log probs for x and y
        recon = self.decoder(z)
        log_qyx = self.classifier_loss(x, y)
        log_pxz = self.img_log_likelihood(recon, x)

        # compute gradients only wrt to params of qyz, no propogating to qzx 
        # see https://arxiv.org/abs/2006.10102 Appendix C.3.1
        log_qyzc_ = dist.Bernoulli(probs=self.classifier(zc.detach())).log_prob(y).sum(dim=-1)
        w = torch.exp(log_qyzc_ - log_qyx)
        if self.mode == 'FC':
            elbo = (w * (log_pxz - kl - log_qyzc) + log_qyx).mean()
        else:
            elbo = (w * ((log_pxz - kl).mean(-1) - log_qyzc) + log_qyx).mean()
        return -elbo

    def classifier_loss(self, x, y, k=100):
        """
        Computes the classifier loss.
        """
        post_params = self.encoder(x)
        zc, _ = dist.Normal(*post_params).rsample(torch.tensor([k])).split([self.z_classify, self.z_style], -1)
        if self.mode == 'FC':
            probs = self.classifier(zc.view(-1, self.z_classify))
        else:
            probs = self.classifier(zc.view(-1, 61, self.z_classify))
        d = dist.Bernoulli(probs=probs)
        y = y.expand(k, -1, -1).contiguous().view(-1, self.num_classes)
        lqy_z = d.log_prob(y).view(k, x.shape[0], self.num_classes).sum(dim=-1)
        lqy_x = torch.logsumexp(lqy_z, dim=0) - np.log(k)
        return lqy_x

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
        zc, _ = dist.Normal(*post_params).rsample(torch.tensor([k])).split([self.z_classify, self.z_style], -1)
        if self.mode == 'FC':
            probs = self.classifier(zc.view(-1, self.z_classify))
        else:
            probs = self.classifier(zc.view(-1, 61, self.z_classify))
        y = y.expand(k, -1, -1).contiguous().view(-1, self.num_classes)
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