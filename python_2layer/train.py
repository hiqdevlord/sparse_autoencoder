# Sparse Autoencoder

# Instructions
# ------------

# This file is a framework for a simple two-layer sparse autoencoder.
# It could be applied in some Machine Learning tasks. The weightes and
# bias learned by Back-propagation could be used to obtain a series of
# features, which taken from the higher hidden layer of local network.

# The multi-layer autoencoder is under construction.

##====================================================================

import numpy as np
from scipy import optimize
from sparse_autoencoder import * 


def main():

    visible_size = 64  # number of input units
    hidden_size = 25    # number of hidden units
    sparsity_param = 0.01   # desired average activation

    lamb = 0.0001     # weight decay parameter
    beta = 3    # weight of sparsity penalty dataset

# Generate training and testing set
    data_train = np.random.rand(64, 1000)  # 1000 samples, with
                                           # dimensionality of 64
    #data_test = np.random.rand(64, 10)

# Obtain random parameters concatenation
    theta = initial_parameter(hidden_size, visible_size)

# Implement sparse autoencoder cost
    options = (data_train, visible_size, hidden_size,
               lamb, sparsity_param, beta)

    opttheta = optimize.fmin_l_bfgs_b(compute_cost, theta,
                                      compute_grad, options)

# Output a .txt file
    fout = open("weights_bias.txt", 'w')
    for op in opttheta[0]:
        #print i
        fout.write(str(op) + '\n')

    print "Mission complete!"


if __name__ == '__main__':
    main()
