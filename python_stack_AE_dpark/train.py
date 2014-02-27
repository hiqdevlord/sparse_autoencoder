# Sparse Autoencoder
# Multi-layer version.

# Instructions
# ------------

# This file is a framework for a stacked version of sparse autoencoder.
# It could be applied in some Machine Learning tasks. The information
# of interest is contained in the activation of the deepest layer of
# hidden units.

##====================================================================

import sys
sys.path.append("./visualization")
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sparse_autoencoder import *
from stack_and_para import vecstack2stack
from visualize import sample_image, display_effect
from dpark import DparkContext
import pickle


def main():

    # Loading data
    print "Loading..."
    data_train = sample_image()

    # Initialize networks
    visible_size = 64  # number of input units
    hidden_size = [25, 16, 9]  # number of hidden units of each layer

    lamb = 0.0001     # weight decay parameter
    beta = 3    # weight of sparsity penalty dataset

    # dpark initialize
    dpark_ctx = DparkContext()

    # Start training, and L-BFGS is adopted
    # We apply a stack-wise greedy training process
    layer_ind = range(len(hidden_size) + 1)
    layer_ind.remove(0)
    layer_size = [visible_size] + hidden_size

    # desired average activation
    sparsity_param = dict()
    for ind in layer_ind:
        # standard: 64 units -> sparsity parameter 0.01
        sparsity_param[ind] = layer_size[ind - 1] * 0.01 / 64

    data = data_train
    opttheta = dict()  # parameter vector of stack AE
    img = dict()  # visualization mode

    for ind in layer_ind:

        print "start training layer No.%d" % ind

        # Obtain random parameters of considered layer
        theta = initial_parameter(layer_size[ind], layer_size[ind - 1])

        # Training begins
        options = (data, layer_size[ind - 1], layer_size[ind],
                   lamb, sparsity_param[ind], beta, dpark_ctx)

        opt = optimize.fmin_l_bfgs_b(compute_cost, theta,
                                     compute_grad, options)

        opttheta[ind] = opt[0]

        W = opttheta.get(ind)[:layer_size[ind]*layer_size[ind-1]].\
            reshape(layer_size[ind], layer_size[ind-1])

        data = np.dot(W, data)

        # visulization shows
        img[ind] = display_effect(W)
        plt.axis('off')
        plt.savefig(str(ind) + '.jpg')

    # Trained parameters of stack AE
    para_stack = vecstack2stack(opttheta, hidden_size, visible_size)

    # Save trained weights and bias
    out = open("weights_bias.pkl", "wb")
    pickle.dump(para_stack, out)
    out.close()

    print "Mission complete!"


if __name__ == '__main__':
    main()
