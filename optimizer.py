import torch.nn.functional as F

def loss_function(preds, labels, mu, logvar):
    # Reconstruction + KL divergence losses summed over all elements and batch
    BCE = F.binary_cross_entropy(preds, labels, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()

    return BCE + KLD