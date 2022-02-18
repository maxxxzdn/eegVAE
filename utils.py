import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist
from scipy.signal import coherence, csd

def define_param():
    param = nn.Parameter(torch.Tensor([0.0]))
    param.requires_grad = True
    return param

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Aggregator(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
    def forward(self, input):
        if self.mode == 'mean':
            return input.mean(1)
        elif self.mode == 'max':
            return input.max(1).values
        elif self.mode == 'none':
            return input.view(input.shape[0],-1)

class UnFlatten(nn.Module):
    def __init__(self, var, size=None):
        super().__init__()
        self.var = var
        self.size = size
    def forward(self, input):
        if self.var == 'z':
            return input.view(input.shape[0]*input.shape[1], -1, self.size)
        elif self.var == 'w':
            return input.view(-1, 16, self.size, self.size)

class ToSymmetric(nn.Module):
    def __init__(self, from_triang):
        super().__init__()
        self.from_triang = from_triang
    def forward(self, input):
        if not self.from_triang:
            input = input.tril(-1)
            return input + input.permute(0,1,3,2)
        else:
            dim = input.shape[-1]                
            output = tensor_from_trian(input.view(-1, dim))
            return output.view(input.shape[0], input.shape[1], output.shape[-1], output.shape[-1])
               
class Diagonal(nn.Module):
    def __init__(self, dim):
        super(Diagonal, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(1+0.1*torch.randn(self.dim))
        self.bias = nn.Parameter(0+0.1*torch.randn(self.dim))

    def forward(self, x):
        return x * self.weight + self.bias
    
class Diagonal_z(nn.Module):
    def __init__(self, dim):
        super(Diagonal_z, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(1 + 0.1*torch.randn(61, self.dim))
        self.bias = nn.Parameter(0 + 0.1*torch.randn(61, self.dim))

    def forward(self, x):
        return x * self.weight + self.bias

def dense_to_sparse(adj):
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        adj (Tensor): The dense adjacency matrix.
     :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = adj.nonzero(as_tuple=True)
    edge_attr = adj[index]

    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])

    return torch.stack(index, dim=0).to(adj.device), edge_attr.to(adj.device)

def train(model, optimizer, loader, mode, device, log = 0):
    model.train(mode = mode)
    for data in loader:
        data.to(device)
        optimizer.zero_grad()
        elbo, acc_y, mse_X, mse_A, kl_z, kl_w, log_qy_xa  = model.elbo(data)
        if mode:
            elbo.backward()
            optimizer.step()
    if log:
        if not mode:
            print("Test")
        print(f"ELBO {elbo.item():.2f}, "
              f"Accuracy y: {list(acc_y.cpu().numpy())} "
              f"MSE X: {mse_X.item():.2f}, "
              f"MSE A: {mse_A.item():.2f}, "
              f"KL Z: {kl_z.item():.2f}, "
              f"KL W: {kl_w.item():.2f}, "
              f"log q(y|X,A): {log_qy_xa.item():.2f}, "
             )
    return log_qy_xa.item()

f = lambda t: -0.1 + 0.00909091*121**t

def get_coh(x, fs):
    x = torch.tensor(x) # to do repeat operations
    N, D = x.shape
    pxy = csd(torch.repeat_interleave(x, repeats = N, dim = 0), x.repeat(N,1), D)[-1].sum(-1).reshape(N,N)
    pxx = np.repeat(csd(x, x, fs)[-1].sum(-1), N).reshape(N,N)
    pyy = np.repeat(csd(x, x, fs)[-1].sum(-1), N).reshape(N,N).T
    return pxy/np.sqrt(pxx*pyy)

def get_plv(x):
    N, D = x.shape
    phi_x = torch.repeat_interleave(torch.tensor(np.angle(np.fft.fft(x))), repeats = N, dim = 0)
    phi_y = torch.tensor(np.angle(np.fft.fft(x))).repeat(N,1)
    return (np.abs(np.exp(-1j*(phi_x-phi_y)).sum(-1))/D).reshape(N,N)

def get_fc(eeg_data, fs = None, method = 'corr', EPS = 1e-3):
    """
    Input: 
        eeg_data (np.array): raw EEG data with N channels and D length
        fs (float): sampling frequency
        method (str): 'lps', 'rcoh', 'plv' or 'corr' - Lagged Phase Synchronization, Real(Coherence), Phase Locking Value, Pearson correlation
        EPS (float): constant for numerical stability 
    Output:
        functional connectivity matrix
    """
    N, D = eeg_data.shape
    if method == 'corr':
        fc_adj = np.corrcoef(eeg_data)
        fc_adj = f(fc_adj) # Highligting strong connections
        fc_adj = fc_adj - np.eye(N,N) #Remove self-self connections
    elif method == 'plv':
        fc_adj = get_plv(eeg_data)
        #fc_adj = f(fc_adj) # Highligting strong connections
        fc_adj = fc_adj - np.eye(N,N) #Remove self-self connections
    else:
        coh = get_coh(eeg_data, fs)
        if method == 'rcoh':
            fc_adj = np.real(coh)
            fc_adj = f(fc_adj) # Highligting strong connections
            fc_adj = fc_adj - np.eye(N,N) #Remove self-self connections
        elif method == 'lps':
            fc_adj = np.abs(np.imag(coh))/np.sqrt(1-np.real(coh)**2 + EPS)
        else:
            raise NotImplementedError
    return fc_adj

def var_interv(model, label_values, ind_to_vary, device, n_interventions = 1000):
    """
    Input: 
        model: trained CGVAE model (CCVAE-AX or CCVAE-A).
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
        locs_pwc_y, scales_pwc_y = model.cond_prior_w(y)
        prior_params_w = (torch.cat([locs_pwc_y, model.zeros_w.expand(bs, -1)], dim=-1), 
                          torch.cat([scales_pwc_y, model.ones_w.expand(bs, -1)], dim=-1))
        pw_y = dist.Normal(*prior_params_w)
        w_ = pw_y.sample()
        w = pw_y.loc
        w[:,ind_to_vary] = w_[:,ind_to_vary]
        pa_w = dist.Normal(*model.decoder_w(w))
    return pa_w.loc.squeeze().var(0)

def highest_connections(var_A, channel_names, n):
    """
    Function to print which connections have the highest variance values
    Input:
        var_A (torch.tensor): N x N matrix where each element is variance of FC value after n_interventions.
        channel_names (numpy.array): array with names of electrodes in the same order as in var_A
        n (int): number of connections to print
    """
    print('Connections with highest variance:')
    for i,pair in enumerate(list(np.dstack(np.unravel_index(np.argsort(var_A.ravel()), (model.n_nodes, model.n_nodes)))[0,-(n*2):,:])[::-1]):
        if i % 2 == 0: # since FC matrices are symmetrical we skip different directions
            print('     ' + str(channel_names[pair]) + 
                  f': {var_A[pair[0], pair[1]].item():.3f}') 
