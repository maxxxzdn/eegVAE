import numpy as np
import torch
import torch.distributions as dist


def diff_interv(model, label_values, ind_to_vary, n_interventions = 1000):
    """
    Input: 
        model: trained CCVAE model
        label_values (torch.tensor): a pair of labels different only along a single dimension.
        ind_to_vary (int): index of the dimension of interest.
        n_interventions (int): number of interventions to perform.
    Output:
        N x N matrix where each element is difference of data reconstructions.
    """
    arr = torch.zeros(n_interventions, 61, 61)
    with torch.no_grad():
        for i in range(n_interventions):
            y = label_values
            locs_pzc_y, scales_pzc_y = model.cond_prior(y.cuda())
            prior_params = (torch.cat([locs_pzc_y, model.zeros.expand(2, -1)], dim=-1), 
                              torch.cat([scales_pzc_y, model.ones.expand(2, -1)], dim=-1))
            pz_y = dist.Normal(*prior_params)
            z = pz_y.sample()
            z_ = z[:,ind_to_vary].clone()
            z[:,:] = z[0,:].unsqueeze(0).repeat(2,1)
            z[:,ind_to_vary] = z_
            z[:,3:] = model.zeros.expand(2, -1)
            x = model.decoder(z)[0].squeeze()
            arr[i] = (x[0] - x[1])[2:-1, 2:-1]
    return np.median(arr, 0)
            
def var_interv(model, label_values, ind_to_vary, device, n_interventions = 1000):
    """
    Input: 
        model: trained CCVAE model
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
        locs_pzc_y, scales_pzc_y = model.cond_prior(y)
        prior_params = (torch.cat([locs_pzc_y, model.zeros.expand(bs, -1)], dim=-1), 
                          torch.cat([scales_pzc_y, model.ones.expand(bs, -1)], dim=-1))
        pz_y = dist.Normal(*prior_params)
        z_ = pz_y.sample()
        z = pz_y.loc
        z[:,ind_to_vary] = z_[:,ind_to_vary]
    return model.decoder(z)[0].squeeze().var(0)