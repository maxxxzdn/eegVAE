import torch
import torch.nn.functional as F
import torch.distributions as dist


def calc_acc(model, loader):
    acc = torch.zeros(0, model.num_classes)
    for data in loader:
        if model.mode == 'EEG':
            xs, ys = data.x.unsqueeze(1).cuda(), data.y.cuda()
            xs = xs[:,:,:,::5]
        else:
            xs, ys =  F.pad(input=data.adj.cuda(), pad=(2, 1, 2, 1), mode='constant', value=0), data.y.cuda()
        acc_ = model.classifier_acc(xs, ys).unsqueeze(0).cpu().detach()
        acc = torch.cat([acc, acc_], 0)
    return acc.mean(0)

def run_model(model, train_loader, test_loader, optim, n_epochs = 100):
    for epoch in range(0, n_epochs):
        model.train()
        for data in train_loader:
            if model.mode == 'EEG':
                xs, ys = data.x.unsqueeze(1).cuda(), data.y.cuda()
            else:
                xs, ys =  F.pad(input=data.adj.cuda(), pad=(2, 1, 2, 1), mode='constant', value=0), data.y.cuda()
            #xs = xs[:,:,:,::5]
            loss = model.elbo(xs,ys)
            loss.backward()
            optim.step()
            optim.zero_grad()
         
        model.eval()
        acc_train = calc_acc(model, train_loader)
        acc_test = calc_acc(model, test_loader)
 
        print('Epoch: {}, Train acc: {}, Test acc: {}'.format(epoch, 
            [round(a.item(),2) for a in acc_train], 
            [round(a.item(),2) for a in acc_test]))
