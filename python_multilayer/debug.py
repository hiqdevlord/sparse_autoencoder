# Instructions
# ------------
# This file helps you to check your BP implementation, which
# is highly recommended to check precisely.

# The diff, given at the last, represent the difference between
# the gradients from BP and numerical computing. It should be
# very small, otherwise the autoencoder was not trained correctly.

##==========================================================


import numpy as np
import sparse_autoencoder as sp
import compute_numerical_grad as co


data = np.random.rand(64, 100)
visible_size = 64
hidden_size = [16, 8, 4]
#hidden_size = [25]
theta = sp.initial_parameter(hidden_size, visible_size)

bp_cost = sp.compute_cost(theta, data, visible_size, hidden_size)
print bp_cost

bp_grad = sp.compute_grad(theta, data, visible_size, hidden_size)
print bp_grad

num_grad = co.compute_numerical_grad(sp.compute_cost, theta, data,
                                     visible_size, hidden_size)

diff = np.linalg.norm(bp_grad - num_grad) / np.linalg.norm(num_grad + bp_grad)

print str(diff) + " should be less than 1e-9! Is it?"
