# eegVAE
Variational Inference on EEG data.
We aim to learn the following latent-variable model: 

<a href="https://www.codecogs.com/eqnedit.php?latex=p(X,A,Z,\omega)&space;=&space;p(\omega)\cdot&space;p(Z)\cdot&space;p(A|\omega)\cdot&space;p(X|Z)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(X,A,Z,\omega)&space;=&space;p(\omega)\cdot&space;p(Z)\cdot&space;p(A|\omega)\cdot&space;p(X|Z)" title="p(X,A,Z,\omega) = p(\omega)\cdot p(Z)\cdot p(A|\omega)\cdot p(X|Z)" /></a>

where X - graph node features, A - connectivity matrix, Z - latent node features, <a href="https://www.codecogs.com/eqnedit.php?latex=w" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w" title="w" /></a> - latent global features.

To do so, we create an inference model of the following form: 

<a href="https://www.codecogs.com/eqnedit.php?latex=q(Z,&space;\omega|X,A)&space;=&space;q(Z|X)\cdot&space;q(\omega|Z)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q(Z,&space;\omega|X,A)&space;=&space;q(Z|X)\cdot&space;q(\omega|Z)" title="q(Z, \omega|X,A) = q(Z|X)\cdot q(\omega|Z)" /></a>

The model is learned further via VAE.

**VAE, cVAE**: double latent variable model and its conditional extension.

**dataset**: routines to create a dataset to train a model on. 

**training, visualize, logger, utils**: training, visualization, logging and other routines.

**main**: script to run.

**metrics**: metrics to assess a model's performance.
