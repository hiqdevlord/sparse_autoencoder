sparse_autoencoder
==================
>**Designer:** Junbo Zhao, Wuhan University, Working at Douban Inc.      
**Email:** zhaojunbo1992chasing@gmail.com      +86-18672365683 

Introduction
-------------------------------------------
      Sparse autoencoder is a great Machine Learning tool in this Deep Learning trend.
    It is proved that this unsupervised tool can be exploited to build far better features than tradictional 
    approaches, like HOGs, LBPs in vision background or MFCC in speech community.           
      To get more infomation on Sparse Autoencoder, please visit:        
      http://deeplearning.net/reading-list/

### As a start
    In this folder, I only implemented a 2 layer framework of Sparse Autoencoder. Note that this is only a 
    start, Back-propagation algorithm and L-BFGS, as the optimization approach, are successfully written or adopted.
    To extend this work, I will commit a multi-layer version in this week, which is easily to derive based on this 
    work. Importantly, Convolution and Softmax will be combined in the future.

### Scipy and Dpark
    I use the "optimize" module of scipy, thus if you don't want to follow my top module script (train.py), you should do:        
    import numpy as np
    from scipy import optimize

    Dpark is in need since this project is compatible to be paralized. About Dpark:        
    https://github.com/douban/dpark

### Platform
    python 2.7.5, numpy 1.6.1 and scipy 0.13.3 on Ubuntu 12.04 LTS.
    
