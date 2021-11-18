import torch
torch.manual_seed(12345)
from torch import nn
import torch.nn.functional as F
import wandb
from dataset import *
from training import *
import argparse

parser = argparse.ArgumentParser(description='eegVAE')
parser.add_argument('--latent_dim_z', type=int, default=10,
                    help='Latent dimension Z')
parser.add_argument('--latent_dim_w', type=int, default=10,
                    help='Latent dimension Omega')
parser.add_argument('--likelihood_a', type=str, default='Normal',
                    help='Normal or CB')
parser.add_argument('--conditional', type=int, default=0,
                    help='conditional or not')
parser.add_argument('--m', type=int, default=1,
                    help='samples per graph')
parser.add_argument('--drop_rate', type=float, default=0.1,
                    help='dropout rate')
parser.add_argument('--alpha_x', type=float, default=1.,
                    help='coefficient for reconstruction_X term')
parser.add_argument('--alpha_a', type=float, default=1.,
                    help='coefficient for reconstruction_A term')
parser.add_argument('--beta_z', type=float, default=1.,
                    help='coefficient for kl_z term')
parser.add_argument('--beta_w', type=float, default=1.,
                    help='coefficient for kl_w term')
parser.add_argument('--wandb', type=int, default=1)
parser.add_argument('--n_nodes', type=int, default=61)
parser.add_argument('--aggr_z', type=str, default='none', help='choose form ["mean", "max", "none"]')
parser.add_argument('--decoder_w_mode', type=str, default='mlp', help='choose form ["mlp", "conv"]')
parser.add_argument('--epochs', type=int, default=50,
                    help='# epochs')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dataset = (torch.load('/data/low_control_corr') + torch.load('/data/low_avh_corr'))
random.shuffle(dataset)
train_dataset = dataset[:8*len(dataset)//10]
test_dataset = dataset[8*len(dataset)//10:]
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)

if args.conditional == 1:
    from cVAE import *
    model = cVAE(args)
elif args.conditional == 0:
    from VAE import *
    model = VAE(args)

if args.wandb:
    wandb.init(project='eegVAE', entity='aipp')
    wandb.config.update(args)
    wandb.watch(model)
    wandb.log({"number of parameters": sum(p.numel() for p in model.parameters())}, step = 0)
    
model.to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
coefs = [args.alpha_x, args.alpha_a, args.beta_z, args.beta_w]

best_elbo = np.inf
for epoch in range(args.epochs):
    train_losses= train(model, optimizer, train_loader, True, device, coefs)
    test_losses = train(model, optimizer, test_loader, False, device, coefs)
        
    if args.wandb:
        if test_losses[0] < best_elbo:
            elbo, recon_loss_x, recon_loss_a, kl_z, kl_w, mse_A = test_losses
            wandb.log({"Best ELBO": elbo,
                       "Best Reconstruction Loss X": recon_loss_x,
                       "Best Reconstruction Loss A": recon_loss_a,
                       "Best KLD Z": kl_z,
                       "Best KLD W": kl_w,
                       "Best MSE A": mse_A,
                       "Best epoch": epoch}, step = epoch)
            best_elbo = test_losses[0]

        elbo, recon_loss_x, recon_loss_a, kl_z, kl_w, mse_A = train_losses
        wandb.log({"Train ELBO": elbo,
                       "Train Reconstruction Loss X": recon_loss_x,
                       "Train Reconstruction Loss A": recon_loss_a,
                       "Train KLD Z": kl_z,
                       "Train KLD W": kl_w,
                       "Train MSE A": mse_A}, step = epoch)
        elbo, recon_loss_x, recon_loss_a, kl_z, kl_w, mse_A = test_losses
        wandb.log({"Test ELBO": elbo,
                       "Test Reconstruction Loss X": recon_loss_x,
                       "Test Reconstruction Loss A": recon_loss_a,
                       "Test KLD Z": kl_z,
                       "Test KLD W": kl_w,
                       "Test MSE A": mse_A}, step = epoch)
