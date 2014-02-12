# Instruction
# -----------
# This file mainly implements Back propagation, and adds weight decay and
# sparsity penalty, which are controlled by lamb and (beta, sparsity_param).
# This file also includes some auxiliary code such as sigmoid function.

##==========================================================
import numpy as np
from stack_and_para import stack2para, para2stack


def initial_parameter(hidden_size, visible_size):
    r = np.sqrt(6) / np.sqrt(sum(hidden_size) + visible_size + 1)

    W = dict()
    b = dict()

    layer_num = len(hidden_size) + 1
    layer_ind = range(layer_num + 1)
    layer_ind.remove(0)

    # Whole network size
    layer_size = [visible_size] + hidden_size + [visible_size]

    # Initialization
    for ind in layer_ind:
        W[ind] = np.random.rand(layer_size[ind], layer_size[ind - 1]) *\
            2 * r - r
        b[ind] = np.zeros(layer_size[ind])

    stack = (W, b)
    # Convert to vector form
    theta = stack2para(stack)

    return theta


def sigmoid(x):
    # Regularly, I choose sigmoid function as the active function.
    if not isinstance(x, np.ndarray):
        print "Wrong parameter of sigmoid function"
        return False
    sigm = 1.0 / (1 + np.exp(-x))
    return sigm


def compute_cost(theta, *args):

    assert(len(args) > 2 and len(args) < 7)

    lamb = 0.0001
    sparsity_param = 0.01
    beta = 3
    data = args[0]
    visible_size = args[1]
    hidden_size = args[2]
    assert(isinstance(hidden_size, list))
    if len(args) > 3:
        lamb = args[3]

    if len(args) > 4:
        sparsity_param = args[4]

    if len(args) > 5:
        beta = args[5]

    # Initialize network layers
    z = dict()  # keys are from 2 to number of layers
    a = dict()  # keys are from 1 to number of layers

    # Get parameters from theta of vector version
    stack = para2stack(theta, hidden_size, visible_size)
    W = stack[0]
    b = stack[1]
    layer_num = len(hidden_size) + 1
    layer_ind = range(layer_num + 1)
    layer_ind.remove(0)

    cost = 0

    # rho is the average activation of hidden layer units
    rho = dict()
    sparsity_mat = dict()
    sparse_kl = dict()
    for ind in layer_ind[:-1]:
        rho[ind + 1] = np.zeros(b.get(ind).shape)
        sparsity_mat[ind + 1] = np.ones(b.get(ind).shape) * sparsity_param

    # paralization, to be updated
    for i in range(data.shape[1]):
        a[1] = data[:, i].reshape(visible_size, 1)
        for l in layer_ind:
            z[l+1] = np.dot(W[l], a[l]) + b[l]
            a[l+1] = sigmoid(z[l+1])
            if l is layer_ind[-1]:
                break
            rho[l+1] += a[l+1]

        cost += sum(np.power(a.get(a.keys()[-1]) - a[1], 2)) / 2

    # Out of sample-loop
    for k in rho.keys():
        rho[k] /= data.shape[1]
        sparse_kl[k] = sparsity_param * np.log(sparsity_mat[k] / rho[k]) +\
            (1 - sparsity_param) * np.log((1 - sparsity_mat[k]) / (1 - rho[k]))

    cost = cost / data.shape[1]
    for ind in layer_ind:
        cost += lamb / 2 * (sum(sum(np.power(W[ind], 2))))
        if ind is not (layer_ind[0] or layer_ind[-1]):
            cost += beta * sum(sparse_kl[ind])

    return cost


def compute_grad(theta, *args):

    assert(len(args) > 2 and len(args) < 7)

    lamb = 0.0001
    sparsity_param = 0.01
    beta = 3
    data = args[0]
    visible_size = args[1]
    hidden_size = args[2]
    assert(isinstance(hidden_size, list))

    if len(args) > 3:
        lamb = args[3]

    if len(args) > 4:
        sparsity_param = args[4]

    if len(args) > 5:
        beta = args[5]

    # Get parameters from theta of vector version
    stack = para2stack(theta, hidden_size, visible_size)
    W = stack[0]
    b = stack[1]
    layer_num = len(hidden_size) + 1
    layer_ind = range(layer_num + 1)
    layer_ind.remove(0)

    # initialize gradients and delta items
    W_grad = dict()
    b_grad = dict()
    W_delta = dict()
    b_delta = dict()
    W_partial_derivative = dict()
    b_partial_derivative = dict()
    for ind in layer_ind:
        W_grad[ind] = np.zeros(W[ind].shape)
        b_grad[ind] = np.zeros(b[ind].shape)
        W_delta[ind] = np.zeros(W[ind].shape)
        b_delta[ind] = np.zeros(b[ind].shape)

    # initialize network layers
    z = dict()  # keys are from 2 to number of layers
    a = dict()  # keys are from 1 to number of layers
    sigma = dict()  # keys are from 2 to number of layers

    # Sparsity pre-feedforward process, to get rho
    # rho is the average activation of hidden layer units
    rho = dict()
    sparsity_mat = dict()
    for ind in layer_ind[:-1]:
        rho[ind + 1] = np.zeros(b.get(ind).shape)
        sparsity_mat[ind + 1] = np.ones(b.get(ind).shape) * sparsity_param
    # paralization, to be updated
    for i in range(data.shape[1]):
        a[1] = data[:, i].reshape(visible_size, 1)
        for l in layer_ind:
            z[l+1] = np.dot(W[l], a[l]) + b[l]
            a[l+1] = sigmoid(z[l+1])
            if l is layer_ind[-1]:
                break
            rho[l+1] += a[l+1]
    for k in rho.keys():
        rho[k] /= data.shape[1]
    # Backpropogation
    # paralization, to be updated
    sparsity_sigma = dict()
    for i in range(data.shape[1]):
        a[1] = data[:, i].reshape(visible_size, 1)
        for l in layer_ind:
            z[l+1] = np.dot(W[l], a[l]) + b[l]
            a[l+1] = sigmoid(z[l+1])
        sigma[layer_ind[-1] + 1] = -(a[1] - a[layer_ind[-1] + 1]) *\
            (a[layer_ind[-1] + 1] * (1 - a[layer_ind[-1] + 1]))
        for l in range(2, layer_ind[-1] + 1)[::-1]:
            sparsity_sigma[l] = -sparsity_mat[l] / rho[l] +\
                (1 - sparsity_mat[l]) / (1 - rho[l])
            sigma[l] = (np.dot(W[l].T, sigma[l + 1]) + beta *
                        sparsity_sigma[l]) * (a[l] * (1 - a[l]))

        for l in layer_ind:
            W_partial_derivative[l] = np.dot(sigma[l + 1], a[l].T)
            b_partial_derivative[l] = sigma[l + 1]
            W_delta[l] += W_partial_derivative[l]
            b_delta[l] += b_partial_derivative[l]

    # gradient computing
    for l in layer_ind:
        W_grad[l] = W_delta[l] / data.shape[1] + lamb * W[l]
        b_grad[l] = b_delta[l] / data.shape[1]

    # return vector version 'grad'
    gradstack = (W_grad, b_grad)
    grad = stack2para(gradstack)

    return grad
