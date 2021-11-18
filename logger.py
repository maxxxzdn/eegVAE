from torch import Tensor

class Tracker():
    def __init__(self):
        self.elbo = 0
        self.recon_loss_x = 0
        self.recon_loss_a = 0
        self.kl_z = 0
        self.kl_w = 0
        self.mse_A = 0
    def update(self, elbo, recon_loss_x, recon_loss_a, kl_z, kl_w, mse_A):
        self.elbo += elbo.item()
        self.recon_loss_x += recon_loss_x.item()
        self.recon_loss_a += recon_loss_a.item()
        self.kl_z += kl_z.item()
        self.kl_w += kl_w.item()
        self.mse_A += mse_A.item()
    def get_mean(self, N):
        self.elbo /= N
        self.recon_loss_x /= N
        self.recon_loss_a /= N
        self.kl_z /= N
        self.kl_w /= N
        self.mse_A /= N
    def get_losses(self):
        return [self.elbo, self.recon_loss_x, self.recon_loss_a, self.kl_z, self.kl_w, self.mse_A]
    
class Log():
    def __init__(self):
        self.elbo = []
        self.recon_loss_x = []
        self.recon_loss_a = []
        self.kl_z = []
        self.kl_w = []
        self.mse_A = []
    def append(self, losses):
        elbo, recon_loss_x, recon_loss_a, kl_z, kl_w, mse_A = losses
        self.elbo.append(elbo)
        self.recon_loss_x.append(recon_loss_x)
        self.recon_loss_a.append(recon_loss_a)
        self.kl_z.append(kl_z)
        self.kl_w.append(kl_w)
        self.mse_A.append(mse_A)

class Logger():
    def __init__(self):
        self.train = Log()
        self.test = Log()
        self.best_epoch = 0
        self.best_test_loss = 0
        self.best_rec_loss = 0
        self.sparseness = 0
    def append(self, train_losses, test_losses):
        self.train.append(train_losses)
        self.test.append(test_losses)