from logger import Logger, Tracker
from torch_geometric.data import DataLoader
from tqdm import tqdm
from numpy import inf
from torch import save
from utils import trian_elements
from metrics import calculate_sparseness, calculate_cov

def train(model, optimizer, loader, mode = True):
    model.train(mode = mode)
    train_tracker = Tracker()
    for data in loader:
        optimizer.zero_grad()
        recovered, loc, logscale, latent = model(data)
            
        adj = data.adj.reshape(-1, model.n_nodes, model.n_nodes)
        adj = trian_elements(adj)
        
        loss, BCE, KLD, l1_loss = model.loss_function(preds=recovered, labels=adj,
                             loc=loc, logscale=logscale, latent=latent)
        train_tracker.update(loss, BCE, KLD, l1_loss)
        loss.backward()
        if mode:
            optimizer.step()
    train_tracker.get_mean(len(loader.dataset))
    return train_tracker.get_losses()

def fit(model, optimizer, train_loader, test_loader, epochs, save_file):
    all_loader = DataLoader(test_loader.dataset, batch_size=10**10, shuffle=False)
    best_loss = inf
    log = Logger()
    for epoch in tqdm(range(epochs), desc="Training for {} epochs".format(epochs)):
        train_losses= train(model, optimizer, train_loader, True)
        test_losses = train(model, optimizer, test_loader, False)
        if test_losses[0] < best_loss:
            best_loss = test_losses[0]
            log.best_test_loss = best_loss
            log.best_epoch = epoch
            log.sparseness = calculate_sparseness(all_loader, model)
            log.cov = calculate_cov(all_loader, model)
            if save_file is not None:
                save(model.state_dict(), save_file)            
        log.append(train_losses, test_losses)
    print("Optimization Finished!")
    print("Model type: " + model.type)
    print('Best epoch: {}'.format(log.best_epoch), ', Best test set loss: {:.4f}'.format(log.best_test_loss))
    print('Sparseness: {:.3f}'.format(log.sparseness))
    print('Cov: {:.3f}'.format(log.cov))
    return log