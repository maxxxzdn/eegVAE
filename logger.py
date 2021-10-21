from torch import Tensor

class Tracker():
    def __init__(self):
        self.loss = 0
        self.BCE = 0
        self.KLD = 0
        self.l1_loss = 0
    def update(self, loss, BCE, KLD, l1_loss):
        self.loss += loss.item()
        self.BCE += BCE.item()
        self.KLD += KLD.item()
        if l1_loss.__class__ is Tensor:
            self.l1_loss += l1_loss.item()
        else: 
            self.l1_loss += l1_loss
    def get_mean(self, N):
        self.loss /= N
        self.BCE /= N
        self.KLD /= N
        self.l1_loss /= N
    def get_losses(self):
        return [self.loss, self.BCE, self.KLD, self.l1_loss]
    
class Log():
    def __init__(self):
        self.loss = []
        self.BCE = []
        self.KLD = []
        self.l1_loss = []
    def append(self, losses):
        loss, BCE, KLD, l1_loss = losses
        self.loss.append(loss)
        self.BCE.append(BCE)
        self.KLD.append(KLD)
        self.l1_loss.append(l1_loss)

class Logger():
    def __init__(self):
        self.train = Log()
        self.test = Log()
        self.best_epoch = 0
        self.best_test_loss = 0
        self.sparseness = 0
    def append(self, train_losses, test_losses):
        self.train.append(train_losses)
        self.test.append(test_losses)