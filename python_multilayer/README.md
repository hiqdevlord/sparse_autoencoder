Sparse Autoencoder
==================
>**Designer:** Junbo Zhao, Wuhan University, Working at Douban Inc.      
**Email:** zhaojunbo1992chasing@gmail.com      +86-18672365683 

Introduction         
-------------------------------------------
  Sparse Autoencoder is a great Machine Learning tool in this Deep Learning trend. It is proved that this unsupervised tool can be exploited to build far better features than tradictional approaches, like HOGs, LBPs in vision background or MFCC in speech community.           
  To get more infomation on Sparse Autoencoder, please visit:        

Multi-layer
-------------------------------------------
  In this folder, the multi-layer framework of Sparse Autoencoder is implemented. Note that in train.py there is only a demo, formed by 3 hidden layers. Back-propagation algorithm and L-BFGS, as the optimization approach, are successfully written or adopted. In the future, Convolution and Softmax will be added in the framework.

Usage
-------------------------------------------
### To get started
  You can simply have a view at train.py, which will show you how to train a Sparse Autoencoder. Importantly, the data matrix should be arranged column-by-column and the size of all layers (visible layers and hidden layers) should be given before training.

### Parameters of training
  lamb: controls the weight decay.
  beta and sparsityparam: control sparsity penalty

### scipy and numpy (if not follow my top module script, train.py)
    import numpy as np
    from scipy import optimize
    
### Dpark
  Dpark is in need since this project is compatible to be paralized. 
  About Dpark: [Dpark](https://github.com/douban/dpark)

### Platform
  python 2.7.5, numpy 1.6.1 and scipy 0.13.3 on Ubuntu 12.04 LTS.
    
