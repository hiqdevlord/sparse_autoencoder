# Instruction
# -----------
# This file can help you convert the weights or bias stack,
# as "dict" in python, to a parameter vector.
# Accordingly, the conversion from vector to stack is as well
# implemented.

# The stacks are constructed as a tuple, combined by weight dict
# & bias dict. Both dicts are formed layer by layer, whose keys
# indicate layer indexes and items are weights or bias of the
# considered layer.
# Usage:
#   W = stack[0]
#   b = stack[1]

# Remember, the hidden layer number is 1 less than the weight &
# bias number!
##=============================================================

import numpy as np


def stack2para(stack):
    # QA
    assert(isinstance(stack, tuple))
    assert(stack[0].keys() == stack[1].keys())
    layer_num = max(stack[0].keys())  # should be len(hidden_size) + 1
    layer_ind = range(layer_num + 1)
    layer_ind.remove(0)
    assert(stack[0].keys() == layer_ind)

    # Conversion
    theta_weight = []
    theta_bias = []
    W = stack[0]
    b = stack[1]
    for k in layer_ind:
        assert(W[k].shape[0] == b[k].shape[0])
        theta_weight = np.hstack((theta_weight, W[k].reshape(W[k].size)))
        theta_bias = np.hstack((theta_bias, b[k].reshape(b[k].size)))

    theta = np.hstack((theta_weight, theta_bias))

    return theta


def para2stack(theta, hidden_size, visible_size):
    assert(isinstance(theta, np.ndarray))
    layer_num = len(hidden_size) + 1
    layer_ind = range(layer_num + 1)
    layer_ind.remove(0)

    # Whole network size
    layer_size = [visible_size] + hidden_size + [visible_size]

    # weight parsing
    weight = dict()
    weight_pos = 0
    for ind in layer_ind:
        weight_pos_pre = weight_pos
        weight_pos += layer_size[ind] * layer_size[ind - 1]
        weight[ind] = theta[weight_pos_pre: weight_pos].\
            reshape(layer_size[ind], layer_size[ind - 1])

    # bias parsing
    bias = dict()
    bias_pos = weight_pos
    for ind in layer_ind:
        bias_pos_pre = bias_pos
        bias_pos += layer_size[ind]
        bias[ind] = theta[bias_pos_pre: bias_pos].\
            reshape(layer_size[ind], 1)

    # stack forming, with a series of tuples
    stack = (weight, bias)

    return stack
