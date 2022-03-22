import numpy as np
from tqdm import tqdm
import torch
import torch.distributions as dist

def run_ccvae(cc_vae, optim, train_loader, test_loader, n_epochs = 100, verbose = True):
    
    it = range(0, n_epochs) if verbose else tqdm(range(0, n_epochs))
        
    for epoch in it:

        batches_per_epoch = len(train_loader)
        train = iter(train_loader)
        test = iter(test_loader)
        cc_vae.train()
        
        for i in range(batches_per_epoch):
            xs, ys = next(train)
            loss = cc_vae.sup(xs, ys)
            loss.backward()
            optim.step()
            optim.zero_grad()
         
        cc_vae.eval()
        acc_train = np.prod(cc_vae.classifier_acc(xs, ys))
        xs, ys = next(test)
        acc_test = np.prod(cc_vae.classifier_acc(xs, ys))
        if verbose:
            print(f"Epoch: {epoch}, "
                  f"Training accuracy: {acc_train:.2f}, "
                  f"Testing accuracy: {acc_test:.2f}")
            
def var_interv(cc_vae, label_values, ind_to_vary, device, n_interventions = 1000):
    """
    Input: 
        cc_vae: trained CCVAE model
        label_values (torch.tensor): tensor of shape [1, #labels] with fixed values of labels.
        ind_to_vary (int): index of dimensions which is to be intervened upon.
        device (str): 'cuda' or 'cpu'.
        n_interventions (int): number of interventions to perform.
    Output:
        N x N matrix where each element is variance of FC value after n_interventions.
    """
    with torch.no_grad():
        bs = n_interventions
        label_values = label_values.expand(n_interventions, -1).clone()
        y_i = torch.randint(2,[n_interventions]) # random binary value
        label_values[:, ind_to_vary] = y_i
        y = label_values.to(device)
        locs_pzc_y, scales_pzc_y = cc_vae.cond_prior(y)
        prior_params = (torch.cat([locs_pzc_y, cc_vae.zeros.expand(bs, -1)], dim=-1), 
                          torch.cat([scales_pzc_y, cc_vae.ones.expand(bs, -1)], dim=-1))
        pz_y = dist.Normal(*prior_params)
        z_ = pz_y.sample()
        z = pz_y.loc
        z[:,ind_to_vary] = z_[:,ind_to_vary]
    return cc_vae.decoder(z).squeeze().var(0)