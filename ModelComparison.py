#!/usr/bin/env python
# coding: utf-8

# In[1]:


from models import *
import models as m
from utils import * 
from optimizer import *
from visualize import *
from meta import *
import random
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import normalized_mutual_info_score as mis
import numpy as np


# ## Initialize dataset

# In[2]:


data_name = "Corr" #"Corr" Mutagenicity" #'IMDB-BINARY' 'MUTAG'


# In[3]:


if data_name == "Corr":
    parameters = {'n_nodes' : 30, 'noise_mu' : 0.5, 'len_sig' : 100, 'phase_mu' : np.pi/10, 'n_active_nodes' : 6}
    active_nodes = [1,2,3,4,14,15,16,27,28,29]
    dataset = CorrDataset(parameters, active_nodes, 200).dataset
else:    
    dataset = TUDataset(root='data/TUDataset', name=data_name)
    
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')


# In[4]:


dataset = [data for data in dataset if data.x.shape[0] < 40]


# In[5]:


max_n_nodes = get_max_n_nodes(dataset)
print(f'Max number of nodes: {max_n_nodes}')


# In[6]:


dataset = add_edge_noise(dataset, 0, 10)


# In[7]:


dataset = update_dataset(dataset, max_n_nodes, None)


# In[8]:


torch.manual_seed(12345)
random.shuffle(dataset)

train_dataset = dataset[:3*len(dataset)//4]
test_dataset = dataset[3*len(dataset)//4:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')


# In[9]:


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# ## Conduct experiments

# In[10]:


latent_dim = 2

n_nodes = max_n_nodes
feat_dim = dataset[0].x.shape[1]
edge_dim = 0 #dataset[0].edge_attr.shape[1]
hidden_dim_ = 64 # adjMLP
hidden_dim = 16 # GNN
beta = 1.
gamma = 0.
delta = 0.
dropout = 0.0
lr = 1e-3
epochs = 200
save_file = 'model'
model_type = 'adj'

prior = Normal(0,1.)
posterior = Normal()


# In[11]:


constants = {'model_type': 'conv_mlp', 'posterior': posterior, 'n_nodes': max_n_nodes, 'feat_dim': feat_dim, 'hidden_dim': hidden_dim, 'latent_dim': latent_dim, 'dropout': dropout, 'beta': beta, 'gamma': gamma, 'delta': delta, 'lr': lr, 'epochs': 500}


# In[12]:


to_vary = {'prior':[Normal(0,1.), Normal(0,0.5), Normal(0,0.1)]}


# In[13]:


parameters = constants | to_vary


# In[20]:


e = Experiment(parameters)


# In[15]:


e.run(train_loader, test_loader, N = 5)

e.visualize('c1_loss')
# In[17]:


e.save()

