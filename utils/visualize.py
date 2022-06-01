import string
import torch
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch.distributions as dist
from mne.viz import plot_connectivity_circle as plot_con, circular_layout


left = ['AF3', 'AF7', 'Fp1', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3', 'FC5', 'FT7', 'FT9', 'T7', 'TP7', 'TP9', 'C1', 'C3', 'C5', 'CP1', 'CP5', 'P1', 'P3', 'P5', 'P7', 'PO3', 'PO9', 'O1']
right = ['AF4', 'AF8', 'Fp2', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T8', 'TP8', 'TP10', 'C2', 'C4', 'C6', 'CP2', 'CP6', 'P2', 'P4', 'P6', 'P8', 'PO4', 'PO10','O2']
upcenter = ['Fz', 'FCz', 'Cz']
botcenter = ['CPz', 'Pz', 'Oz', 'POz']

def visualize_interv(path, fc_diff, vmin = -0.3, vmax = 0.3):
    pos = np.loadtxt(path, str)[:,:3].astype(float)
    names = np.loadtxt(path, str)[:,3].copy()
    order = np.array([np.where(names == np.array(upcenter + left + botcenter + right[::-1])[i]) for i in range(61)]).reshape(-1)
    names = names[order]
    node_angles = circular_layout(names, list(names), start_pos=77,
                                  group_boundaries=[0, len(upcenter), len(upcenter) + len(left), len(upcenter) + len(left) + len(botcenter)])
    con1 = fc_diff.copy()
    con2 = fc_diff.copy()
    mask1 = (con1 > np.sort(con1.reshape(-1))[50])
    mask2 = (con2 < np.sort(con2.reshape(-1))[-50])
    con1[mask1] = 0
    con2[mask2] = 0
    con = con1 + con2
    con = con[order].T[order].T
    fig = plt.figure(figsize = [15,15])
    plot_con(con[:,:], names, node_angles = node_angles, vmin = vmin, vmax = vmax, colormap='coolwarm', facecolor='white', textcolor='black', fontsize_names = 24, colorbar_size = 0.5, fontsize_colorbar = 24, colorbar_pos = (-0.75, 0.5), linewidth=10, fig=fig)

def visualize_latent_2d(model, x, dim1, dim2):
    model = model.eval()
    qz_x = dist.Normal(*model.encoder(x))
    z_post = qz_x.sample([2]).view(-1, model.z_dim)
    
    u = 30*torch.rand((5000,10))-15
    df1 = pd.DataFrame(np.array(u))

    df2 = pd.DataFrame(np.array(z_post.cpu().detach()))
    df = pd.concat([df1,df2], join="inner", ignore_index=True)\

    sns.set_style("whitegrid")
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sns.set_context("paper") 

    sns.kdeplot(
        data=df, x=dim1, y=dim2, cmap="plasma", fill=True,
    )
    plt.xlim(-8,8)
    plt.ylim(-8,8)
    plt.xlabel('Latent variable $\mathbf{z}^{schizophrenia}_c$', fontsize=20)
    plt.ylabel('Latent variable $\mathbf{z}^{hallucinations}_c$', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

def visualize_latent(model, x, y, label_names, k = 100, epoch = None):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sns.set_context("paper", rc={"xtick.labelsize":32,"axes.labelsize":48}) 

    zc_letters = [str(i) for i in range(100)][:model.z_classify]
    zs_letters = [str(i) for i in range(100)][model.z_classify:(model.z_style+model.z_classify)]

    y_unique = y.unique(dim = 0)
    locs_pwc_y, scales_pwc_y = model.cond_prior(y_unique)
    prior_params_w = (torch.cat([locs_pwc_y, model.zeros.expand(len(y_unique), -1)], dim=-1), 
                      torch.cat([scales_pwc_y, model.ones.expand(len(y_unique), -1)], dim=-1))
    
    pw_y = dist.Normal(*prior_params_w)
    qw_a = dist.Normal(*model.encoder(x))

    w_post = qw_a.sample([k]).flip(-1).reshape(-1).cpu().detach().numpy() 
    w_prior = pw_y.sample([len(w_post)]).flip(-1).reshape(-1).cpu().detach().numpy()[:len(w_post)]
    g = np.tile((zc_letters + zs_letters)[::-1], k*len(y))
    df = pd.DataFrame(dict(x=w_post, y=w_prior, g=g))
    df = df[df.g.astype(int) < 5]
    df.g = df.g.replace(zc_letters, label_names)

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=1.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "y", clip_on=(-15, 15),
          fill=True, alpha=.5, linewidth=1.5, color = 'r'),  #, bw_adjust=.1)
    g.map(sns.kdeplot, "x", clip_on=(-15, 15),
          fill=True, alpha=.8, linewidth=1.5) #, bw_adjust=.1)
    g.map(sns.kdeplot, "x", clip_on=(-15, 15), color="w", lw=2) #, bw_adjust=.1)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=(-15, 15))


    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .4, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes, fontsize = 32)


    g.map(label, "x")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.4)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set(xlabel="$\omega$")
    g.despine(bottom=True, left=True)

    g.set(xlim=(-15, 15))
    if epoch is not None:
        ax = plt.gca()
        ax.text(0.8, 3, 'epoch: ' + str(epoch), fontweight="bold",
                ha="left", va="center", transform=ax.transAxes, fontsize = 32)
        fig = g.fig
        fig.savefig('./plots/latent_space' + str(epoch), bbox_inches='tight')
        
    return df