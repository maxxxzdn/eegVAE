import torch
from logger import Logger, Tracker
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from numpy import inf
from torch import save
from utils import trian_elements, scheduler
from metrics import calculate_sparseness, calculate_cov

def train(model, optimizer, loader, mode, device, coefs):
    model.train(mode = mode)
    train_tracker = Tracker()
    for data in loader:
        data.to(device)
        optimizer.zero_grad()
        x_hat, a_hat, z, w, loc_z, scale_z, loc_w, scale_w = model(data)
        args = [data.x, data.adj, x_hat, a_hat, z, w, loc_z, scale_z, loc_w, scale_w]
        elbo, recon_loss_x, recon_loss_a, kl_z, kl_w = model.loss_function(args, coefs)
        mse_A = F.mse_loss(data.adj.unsqueeze(0), a_hat, reduction = 'none').sum(-1).sum(-1).mean()
        train_tracker.update(elbo, recon_loss_x, recon_loss_a, kl_z, kl_w, mse_A)
        if mode:
            elbo.backward()
            optimizer.step()
    train_tracker.get_mean(len(loader))
    return train_tracker.get_losses()

def fit(model, optimizer, train_loader, test_loader, epochs, device, coefs):
    all_loader = DataLoader(test_loader.dataset, batch_size=10**10, shuffle=False)
    best_loss = inf
    log = Logger()
    for epoch in tqdm(range(epochs), desc="Training for {} epochs".format(epochs)):
        train_losses = train(model, optimizer, train_loader, True, device, coefs)
        test_losses = train(model, optimizer, test_loader, False, device, coefs)
        if test_losses[0] < best_loss:
            best_loss = test_losses[0]
            log.best_test_loss = best_loss
            log.best_rec_loss = test_losses[1]
            log.best_epoch = epoch
            torch.save(model.state_dict(), 'my_model')
        log.append(train_losses, test_losses)
    print("Optimization Finished!")
    print('Best epoch: {}'.format(log.best_epoch), ', Best test rec loss: {:.4f}'.format(log.best_rec_loss), ', Best test loss: {:.4f}'.format(log.best_test_loss))
    return log

"""

def train(model, optimizer, loader, device, mode = True, beta=1.):
    model.train(mode = mode)
    train_tracker = Tracker()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        recovered, loc, logscale, latent = model(data)
            
        adj = data.adj.reshape(-1, model.n_nodes, model.n_nodes)
        adj = trian_elements(adj)
        
        loss, rec, KLD, l1_loss = model.loss_function(preds=recovered, labels=adj,
                             loc=loc, logscale=logscale, latent=latent, beta=beta)
        train_tracker.update(loss, rec, KLD, l1_loss)
        loss.backward()
        if mode:
            optimizer.step()
    train_tracker.get_mean(len(loader.dataset))
    return train_tracker.get_losses()

def fit(model, optimizer, train_loader, test_loader, epochs, save_file, schedule_beta, device):
    all_loader = DataLoader(test_loader.dataset, batch_size=10**10, shuffle=False)
    best_loss = inf
    log = Logger()
    for epoch in tqdm(range(epochs), desc="Training for {} epochs".format(epochs)):
        beta = scheduler(epoch, model.beta, epochs) if schedule_beta else model.beta
        train_losses= train(model, optimizer, train_loader, device, True, beta)
        test_losses = train(model, optimizer, test_loader, device, False, 1.)
        if test_losses[0] < best_loss:
            best_loss = test_losses[0]
            log.best_test_loss = best_loss
            log.best_rec_loss = test_losses[1]
            log.best_epoch = epoch
            log.sparseness = calculate_sparseness(all_loader, model, device)
            log.cov = calculate_cov(all_loader, model, device)
            if save_file is not None:
                save(model.state_dict(), save_file)            
        log.append(train_losses, test_losses)
    print("Optimization Finished!")
    print("Model type: " + model.type)
    print('Best epoch: {}'.format(log.best_epoch), ', Best test rec loss: {:.4f}'.format(log.best_rec_loss), ', Best test loss: {:.4f}'.format(log.best_test_loss))
    
    print('Sparseness: {:.3f}'.format(log.sparseness))
    print('Cov: {:.3f}'.format(log.cov))
    return log
    
"""