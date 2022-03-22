import string
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.distributions as dist

def visualize_latent(cc_vae, x, y, label_names, k = 100, n_priors = 5):
    
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sns.set_context("paper", rc={"xtick.labelsize":32,"axes.labelsize":48}) 
    
    zc_letters = [str(i) for i in range(100)][:cc_vae.z_classify]
    zs_letters = [str(i) for i in range(100)][cc_vae.z_classify:(cc_vae.z_style+cc_vae.z_classify)]

    y_unique = y.unique(dim = 0)
    locs_pzc_y, scales_pzc_y = cc_vae.cond_prior(y_unique)
    prior_params = (torch.cat([locs_pzc_y, cc_vae.zeros.expand(len(y_unique), -1)], dim=-1), 
                      torch.cat([scales_pzc_y, cc_vae.ones.expand(len(y_unique), -1)], dim=-1))
    
    pz_y = dist.Normal(*prior_params)
    qz_x = dist.Normal(*cc_vae.encoder(x))

    z_post = qz_x.sample([k]).flip(-1).reshape(-1).cpu().detach().numpy() 
    z_prior = pz_y.sample([len(z_post)]).flip(-1).reshape(-1).cpu().detach().numpy()[:len(z_post)]
    g = np.tile((zc_letters + zs_letters)[::-1], k*len(y))
    df = pd.DataFrame(dict(x=z_post, y=z_prior, g=g))
    df = df[df.g.astype(int) < n_priors]
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
    