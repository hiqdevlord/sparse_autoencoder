# Instructions
# ------------
# This file helps you to check your BP implementation, which
# is highly recommended to check precisely

##==========================================================

import numpy as np
import sparse_autoencoder as sp
import compute_numerical_grad as co

# Randomly generate theta and data
visible_size = 32
hidden_size = [16, 8, 4]
data = np.random.rand(32, 100)

layer_ind = range(len(hidden_size) + 1)
layer_ind.remove(0)
layer_size = [visible_size] + hidden_size

# Debugging!
for ind in layer_ind:

    theta = sp.initial_parameter(layer_size[ind], layer_size[ind - 1])

    bp_grad = sp.compute_grad(theta, data, layer_size[ind-1], layer_size[ind])

    num_grad = co.compute_numerical_grad(sp.compute_cost, theta, data,
                                         layer_size[ind-1], layer_size[ind])

    diff = np.linalg.norm(bp_grad - num_grad) /\
        np.linalg.norm(num_grad + bp_grad)

    print str(diff) + " should be less than 1e-9! Is it?"

    W = theta[:layer_size[ind]*layer_size[ind-1]].\
        reshape(layer_size[ind], layer_size[ind-1])

    data = np.dot(W, data)
