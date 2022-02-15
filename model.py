import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

import networks
       
class ccVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_nodes = args.n_nodes
        self.x_dim = args.x_dim
        self.use_cuda = args.use_cuda
        self.elbo_coefs = args.elbo_coefs
        
        self.z_dim = args.z_dim
        self.z_classify = args.n_classes
        self.z_style = args.z_dim - self.z_classify
        
        self.w_dim = args.w_dim
        self.w_classify = args.n_classes
        self.w_style = args.w_dim - self.w_classify
        
        self.encoder_x = networks.Encoder_x(args.x_dim, args.n_nodes, args.z_dim)
        self.encoder_a = networks.Encoder_a(args.n_nodes, args.w_dim)       
        self.decoder_za = networks.Decoder_za(args.x_dim, args.n_nodes, args.z_dim)
        self.decoder_w = networks.Decoder_w(args.w_dim, args.n_nodes, args.decoder_w_mode)
        self.cond_prior_w = networks.CondPrior_w(self.w_classify, args.cond_prior_w_mode)
        self.cond_prior_z = networks.CondPrior_Z(args.n_nodes, self.z_classify, args.cond_prior_z_mode)
        self.classifier = networks.Classifier(args.n_nodes, args.n_classes)
                          
        self.zeros_z = torch.zeros(1, self.z_style)
        self.ones_z = torch.ones(1, self.z_style)
        self.zeros_w = torch.zeros(1, self.w_style)
        self.ones_w = torch.ones(1, self.w_style)
        
        if self.use_cuda:  
            self.zeros_z = self.zeros_z.to('cuda')
            self.ones_z = self.ones_z.to('cuda')
            self.zeros_w = self.zeros_w.to('cuda')
            self.ones_w = self.ones_w.to('cuda')
           
    def reconstruct_a(self, a):
        w = dist.Normal(*self.encoder_a(a)).rsample()
        return self.decoder_w(w)
        
    def reconstruct_x(self, data):
        return self.decoder_za(dist.Normal(*self.encoder_x(data.x)).rsample(), data.adj)
    
    def reconstruct_data(self, data):
        return self.reconstruct_x(data), self.reconstruct_a(data.adj)
        
    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def classifier_loss(self, data, k=100):
        """
        Computes the classifier loss.
        """        
        zc, _ = dist.Normal(*self.encoder_x(data.x)).rsample(torch.tensor([k])).split([self.z_classify, self.z_style], -1)
        wc, _ = dist.Normal(*self.encoder_a(data.adj)).rsample(torch.tensor([k])).split([self.w_classify, self.w_style], -1)
        d = dist.ContinuousBernoulli(probs = self.classifier(zc, wc))
        y = data.y.expand(k, -1, -1,).contiguous()
        lqy_zw = (d.log_prob(y)).view(k, data.adj.shape[0], -1).sum(dim=-1)
        lqy_xa = torch.logsumexp(lqy_zw, dim=0) - math.log(k)
        return lqy_xa
            
    def elbo(self, data):
        bs = data.adj.shape[0]
        #Z ~ q(Z|X)
        qz_x = dist.Normal(*self.encoder_x(data.x))
        z = qz_x.rsample([1])  
        #Z = {Z_c, Z_\c}
        zc, zs = z.split([self.z_classify, self.z_style], -1)
        #w ~ q(w|A)
        qw_a = dist.Normal(*self.encoder_a(data.adj))
        w = qw_a.rsample([1])
        #w = {w_c, w_\c}
        wc, ws = w.split([self.w_classify, self.w_style], -1)
        #log q(y|Z_c, w_c),  log q(y|X, A)
        qy_zc_wc = dist.ContinuousBernoulli(probs = self.classifier(zc, wc))
        y = qy_zc_wc.rsample()
        log_qy_zc_wc = qy_zc_wc.log_prob(y)
        log_qy_xa = self.classifier_loss(data)
        #log p(y)
        log_py = dist.ContinuousBernoulli(data.y).log_prob(y)
        
        #elbo_z
        px_za = dist.Normal(*self.decoder_za(z, data.adj))
        log_px_za = px_za.log_prob(data.x.view(bs,self.n_nodes,self.x_dim))
        log_qz_x = qz_x.log_prob(z)
        
        locs_pzc_y, scales_pzc_y = self.cond_prior_z(data.y)
        prior_params_z = (torch.cat([locs_pzc_y, self.zeros_z.expand(bs, self.n_nodes, -1)], dim=-1), 
                          torch.cat([scales_pzc_y, self.ones_z.expand(bs, self.n_nodes, -1)], dim=-1))
        
        log_pz_y = dist.Normal(*prior_params_z).log_prob(z)
        
        kl_z = (log_qz_x - log_pz_y).mean(0).sum(-1).sum(-1)
        recon_x = log_px_za.sum(-1).sum(-1)
        elbo_z = recon_x - self.elbo_coefs[0]*kl_z
        
        #elbo_w
        pa_w = dist.Normal(*self.decoder_w(w))
        log_pa_w = pa_w.log_prob(data.adj)
        log_qw_a = qw_a.log_prob(w)
        
        locs_pwc_y, scales_pwc_y = self.cond_prior_w(data.y)
        prior_params_w = (torch.cat([locs_pwc_y, self.zeros_w.expand(bs, -1)], dim=-1), 
                          torch.cat([scales_pwc_y, self.ones_w.expand(bs, -1)], dim=-1))
        
        log_pw_y = dist.Normal(*prior_params_w).log_prob(w)
        
        kl_w = (log_qw_a - log_pw_y).mean(0).sum(-1)
        recon_a = log_pa_w.sum(-1).sum(-1)
        elbo_w = recon_a - self.elbo_coefs[1]*kl_w
                
        # ELBO
        log_qy_zc_wc_ = (dist.ContinuousBernoulli(probs = self.classifier(zc.detach(), wc.detach())).log_prob(data.y)).mean(0).sum(-1)
        weight = 1 #torch.exp(log_qy_zc_wc_ - log_qy_xa)
        elbo = weight * (elbo_z + elbo_w + (log_py - log_qy_zc_wc).mean(0).sum(-1)) + self.elbo_coefs[2]*log_qy_xa
        
        # accuracies
        y_pred = (qy_zc_wc.probs > 0.5).float().mean(0)
        y_true = data.y
        acc_y = (y_pred == y_true).float().mean(0)
        
        # mse
        mse_X = torch.pow(data.x.view(bs, self.n_nodes, self.x_dim) - px_za.loc.view(bs, self.n_nodes, self.x_dim), 2).sum(-1).sum(-1).mean()
        mse_A = torch.pow(data.adj.view(bs, self.n_nodes, self.n_nodes) - pa_w.loc.view(bs, self.n_nodes, self.n_nodes), 2).sum(-1).sum(-1).mean()
                        
        return -elbo.mean(), acc_y, mse_X, mse_A, kl_z.mean(), kl_w.mean(), log_qy_xa.mean()