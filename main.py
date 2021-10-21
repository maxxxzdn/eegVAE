import random
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse

from models import *
from utils import * 
from visualize import *
from dataset import *
from training import *
import models as m

torch.manual_seed(12345)

parser = argparse.ArgumentParser(description='eegVAE')
parser.add_argument('--data_name', type=str, default='Corr',
                    help='Dataset name from [MUTAG, Mutagenicity, Corr]')
parser.add_argument('--n_graphs', type=int, default=200,
                    help='Number of experiments to train on')
parser.add_argument('--n_noise', type=int, default=0,
                    help='Number of graphs to add')
parser.add_argument('--m_noise', type=int, default=10,
                    help='Number of noisy edges to add')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size')
parser.add_argument('--latent_dim', type=int, default=16,
                    help='Latent dimension')
parser.add_argument('--hidden_dim', type=int, default=16,
                    help='Latent dimension')
parser.add_argument('--beta', type=float, default=1.,
                    help='Beta')
parser.add_argument('--gamma', type=float, default=0.,
                    help='Gamma')
parser.add_argument('--dropout', type=float, default=0.,
                    help='Dropout possibility')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')
parser.add_argument('--epochs', type=int, default=200,
                    help='# epochs to train')
parser.add_argument('--model_type', type=str, default='conv_mlp',
                    help='Type of encoder to use in model from [conv_mlp, adj]')
parser.add_argument('--prior_scale', type=float, default=1.,
                    help='Variance or b for a prior distribution')
parser.add_argument('--prior_family', type=str, default='Normal',
                    help='Distribution family of prior and posterior (Normal or Laplace)')
parser.add_argument('--wandb', type=int, default=0,
                    help='Enable W&B logging')
args = parser.parse_args()

if args.wandb:
    import wandb
    wandb.init(project='eegVAE', entity='aipp')
    wandb.config.update(args)  # adds all of the arguments as config variable

if args.data_name == "Corr":
    parameters = {'n_nodes' : 30, 'noise_std' : 0.5, 'len_sig' : 100, 'phase_std' : np.pi/10, 'n_active_nodes' : 6}
    active_nodes = [1,2,3,4,14,15,16,27,28,29]
    dataset = CorrDataset(parameters, active_nodes, args.n_graphs).dataset
else:    
    dataset = TUDataset(root='data/TUDataset', name=args.data_name)
    
max_n_nodes = get_max_n_nodes(dataset)

dataset = add_edge_noise(dataset, args.n_noise, args.m_noise)
dataset = update_dataset(dataset, max_n_nodes)
random.shuffle(dataset)

train_dataset = dataset[:3*len(dataset)//4]
test_dataset = dataset[3*len(dataset)//4:]

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

if args.prior_family == 'Normal':
    prior = torch.distributions.normal.Normal(0, args.prior_scale)
elif args.prior_family == 'Laplace':
    prior = torch.distributions.laplace.Laplace(0, args.prior_scale)

encoder_ = getattr(m, 'Encoder_{}'.format(args.model_type))
encoder = encoder_(max_n_nodes, dataset[0].x.shape[1], args.hidden_dim, args.latent_dim, args.dropout)
decoder = Decoder(max_n_nodes, args.latent_dim, args.hidden_dim, args.dropout)
model = VAE(encoder, decoder, prior, args.beta, args.gamma) 
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.wandb:
    wandb.watch(model)

_test_loader = DataLoader(test_loader.dataset, batch_size=10**10, shuffle=False)
best_loss = np.inf
for epoch in range(args.epochs):
    train_losses= train(model, optimizer, train_loader, True)
    test_losses = train(model, optimizer, test_loader, False)
    if test_losses[0] < best_loss:
        best_loss, best_BCE, best_KLD, best_l1_loss = test_losses
        if args.wandb:
            wandb.log({"Best Loss": best_loss,
                       "Best BCE": best_BCE,
                       "Best KLD": best_KLD,
                       "Best L1 loss": best_l1_loss,
                       "Best epoch": epoch,
                       "Sparseness": calculate_sparseness(_test_loader, model),
                       "Covariance": calculate_cov(_test_loader, model)}, step = epoch)
        
    if args.wandb:
        loss, BCE, KLD, l1_loss = train_losses
        wandb.log({"Train Loss": loss,
                   "Train BCE": BCE,
                   "Train KLD": KLD,
                   "Train L1 loss": l1_loss}, step = epoch)
        loss, BCE, KLD, l1_loss = test_losses
        wandb.log({"Test Loss":loss,
                   "Test BCE": BCE,
                   "Test KLD": KLD,
                   "Test L1 loss": l1_loss}, step = epoch)