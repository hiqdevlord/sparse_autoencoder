# Sparse Autoencoder
# Multi-layer version.

# Instructions
# ------------

# This file is a framework for a multi-layer sparse autoencoder.
# It could be applied in some Machine Learning tasks. The weightes and
# bias learned by Back-propagation could be used to obtain a series of
# features, which taken from the higher hidden layer of local network.

##====================================================================

import numpy as np
from scipy import optimize
from sparse_autoencoder import * 


def main():

    visible_size = 128  # number of input units
    hidden_size = [64, 32, 16]  # number of hidden units of each layer
    sparsity_param = 0.01   # desired average activation

    lamb = 0.0001     # weight decay parameter
    beta = 3    # weight of sparsity penalty dataset

# Generate training and testing set
    data_train = np.random.rand(128, 1000)  # 1000 samples, with
                                           # dimensionality of 128
    #data_test = np.random.rand(128, 10)

# Obtain random parameter concatenated vector
    theta = initial_parameter(hidden_size, visible_size)

# Start training, and L-BFGS is adopted.
    options = (data_train, visible_size, hidden_size,
               lamb, sparsity_param, beta)

    opttheta = optimize.fmin_l_bfgs_b(compute_cost, theta,
                                      compute_grad, options)

# Output a .txt file
    fout = open("weights_bias.txt", 'w')
    for op in opttheta[0]:
        fout.write(str(op) + '\n')

    print "Mission complete!"


if __name__ == '__main__':
    main()
