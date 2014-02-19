Stacked Sparse Autoencoders
===========================
>**Designer:** Junbo Zhao, Wuhan University, Working at Douban Inc.      
**Email:** zhaojunbo1992chasing@gmail.com      +86-18672365683 

Introduction
------------------------
  Sparse Autoencoders can be stacked in a greedy layerwise fashion for pretraining usage. For instance, MFCC is a well-known feature design in speech community during and its great power lasted for nearly 30 years. However, MFCC is not optimal in another audio community, MIR (Music Information Retrieval). Thus, we need Stacked AE to help us design a new feature automatically.
  To get more information, please visit:
  [UFLDL Tutorial][http://ufldl.stanford.edu/wiki/index.php/Stacked_Autoencoders]

"Stacked"
-----------------------
  A stacked neural network consisting of multiple layers of Autoencoders. Note that "stacked" is not simply "multi-layer" version, which I pushed up a few days ago, although they both have multiple layers. Stacked Autoencoders consist of lots of AEs and each of them only has one hidden layer! On the contrary, multi-layer AE has only one AE, but this single AE has several hidden layers.
  Theoretically, Stacked AE is better than multi-layer AE because the BP converges much more easily of the former than the latter.

Visualization
-----------------------
  We visualize the effect of stacked AE layer-by-layer. Examples are showed in the following, and these images are located in visualization/examples. These three images can help you to understand what it has learned. We obtain them layer-by-layer, applying **Lagrange Multiplier** after imposing a norm constrain. We can find the features that can maximally activate the hidden units and thus, these features are what hidden layers look for.
  
  The three images:
  ![layer 1](https://github.com/zhaojunbo/sparse_autoencoder/tree/raw/python_stack_AE/visualization/examples/1.jpg)
  ![layer 2](https://github.com/zhaojunbo/sparse_autoencoder/tree/raw/python_stack_AE/visualization/examples/2.jpg)
  ![layer 3](https://github.com/zhaojunbo/sparse_autoencoder/tree/raw/python_stack_AE/visualization/examples/3.jpg)

  Accordingly, we can find that Stacked AE could capture some useful features. The first layer tends to learn edges in an input images, and the second layer seems aim at a second-order features, in terms of what edges tend to occur together. The higher layers of the Stacked AE tend to learn higher-order features.

Platform
-------------
  python 2.7.5, numpy 1.6.1, scipy 0.13.3 and matplotlib 1.1.1rc on Ubuntu 12.04 LTS

