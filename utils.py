import torch
import torch.nn as nn

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
