# Instruction
# -----------
# This file mainly implements Back propagation, and adds weight decay and
# sparsity penalty, which are controlled by lamb and (beta, sparsity_param).
# This file also includes some auxiliary code such as sigmoid function.

##==========================================================
import numpy as np
from stack_and_para import stack2vecstack


def initial_parameter(hidden_size, visible_size):

    r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    W1 = np.random.rand(hidden_size, visible_size) * 2 * r - r
    W2 = np.random.rand(visible_size, hidden_size) * 2 * r - r

    b1 = np.zeros(hidden_size)
    b2 = np.zeros(visible_size)

    # Convert weights and bias to the vector form
    theta = np.hstack(([], W1.reshape(W1.size),
                      W2.reshape(W2.size), b1, b2))

    return theta

    '''
    # One-step initialize whole framework
    assert(isinstance(hidden_size, dict))
    layer_ind = range(len(hidden_size) + 1)
    layer_ind.remove(0)
    layer_size = [visible_size] + hidden_size

    W = dict()
    b = dict()

    # Initialization
    for ind in layer_ind:
        W[ind] = dict()
        b[ind] = dict()
        r = np.sqrt(6) / np.sqrt(layer_size[ind] + layer_ind[ind-1] + 1)
        W[ind][1] = np.random.rand(layer_size[ind], layer_size[ind-1]) *\
            2 * r - r
        W[ind][2] = np.random.rand(layer_size[ind-1], layer_size[ind]) *\
            2 * r - r
        b[ind][1] = np.zeros(layer_size[ind])
        b[ind][2] = np.zeros(layer_size[ind-1])

    stack = (W, b)

    # Convert to vector form
    theta = stack2vecstack(stack)

    return theta
    '''


def sigmoid(x):
    # Regularly, I choose sigmoid function as the active function.
    if not isinstance(x, np.ndarray):
        print "Wrong parameter of sigmoid function"
        return False
    sigm = 1.0 / (1 + np.exp(-x))
    return sigm


def compute_cost(theta, *args):

    assert(len(args) > 2 and len(args) < 8)

    lamb = 0.0001
    sparsity_param = 0.01
    beta = 3
    data = args[0]
    visible_size = args[1]
    hidden_size = args[2]

    if len(args) > 3:
        lamb = args[3]

    if len(args) > 4:
        sparsity_param = args[4]

    if len(args) > 5:
        beta = args[5]

    if len(args) > 6:
        dpark = args[6]

    # Initialize network layers
    z = dict()  # keys are from 2 to number of layers
    a = dict()  # keys are from 1 to number of layers

    # Get parameters from theta of vector version
    W1 = theta[: hidden_size * visible_size].reshape(hidden_size, visible_size)
    W2 = theta[hidden_size * visible_size: 2 * hidden_size * visible_size].\
        reshape(visible_size, hidden_size)
    b1 = theta[2 * hidden_size * visible_size: 2 * hidden_size * visible_size +
               hidden_size].reshape(hidden_size, 1)
    b2 = theta[2 * hidden_size * visible_size + hidden_size:].\
        reshape(visible_size, 1)

    # for loop version
    '''
    cost = 0

    # rho is the average activation of hidden layer units
    rho = np.zeros(b1.shape)

    # paralization, to be updated
    for i in range(data.shape[1]):
        a[1] = data[:, i].reshape(visible_size, 1)
        z[2] = np.dot(W1,  a[1]) + b1
        a[2] = sigmoid(z[2])
        rho += a[2]
        z[3] = np.dot(W2, a[2]) + b2
        a[3] = sigmoid(z[3])

        cost += sum(np.power(a[3] - a[1], 2)) / 2
    '''

    # dpark version
    cost_acc = dpark.accumulator(0)

    def map_iter(dat):
        a[1] = dat.reshape(visible_size, 1)
        z[2] = np.dot(W1,  a[1]) + b1
        a[2] = sigmoid(z[2])
        z[3] = np.dot(W2, a[2]) + b2
        a[3] = sigmoid(z[3])

        rho_iter = a[2].reshape(a[2].size)
        cost_acc.add(np.sum(np.power(a[3] - a[1], 2)) / 2)

        return rho_iter

    #print "compute_cost rho collecting"
    rho_collect = dpark.makeRDD(
        data.T, 300
        ).map(
        map_iter
        ).collect()
    rho_collect = np.array(rho_collect).T
    rho = np.sum(rho_collect, 1).reshape(rho_collect.shape[0], 1)
    cost = cost_acc.value
    # dpark finished

    # Out of loop
    rho /= data.shape[1]
    sparse_kl = sparsity_param * np.log(sparsity_param / rho) +\
        (1 - sparsity_param) * np.log((1 - sparsity_param) / (1 - rho))

    '''
    try:
        sparse_kl = sparsity_param * np.log(sparity_param / rho) +\
            (1 - sparsity_param) * np.log((1 - sparsity_param) / (1 - rho))
    except RuntimeWarning:
        print rho
        input()
    '''

    cost = cost / data.shape[1]
    cost += lamb / 2 * (sum(sum(np.power(W1, 2))) + sum(sum(np.power(W2, 2))))
    cost += beta * sum(sparse_kl)

    return cost


def compute_grad(theta, *args):

    assert(len(args) > 2 and len(args) < 8)

    lamb = 0.0001
    sparsity_param = 0.01
    beta = 3
    data = args[0]
    visible_size = args[1]
    hidden_size = args[2]

    if len(args) > 3:
        lamb = args[3]

    if len(args) > 4:
        sparsity_param = args[4]

    if len(args) > 5:
        beta = args[5]

    if len(args) > 6:
        dpark = args[6]

    # Get parameters from theta of vector version
    W1 = theta[: hidden_size * visible_size].reshape(hidden_size, visible_size)
    W2 = theta[hidden_size * visible_size: 2 * hidden_size * visible_size].\
        reshape(visible_size, hidden_size)
    b1 = theta[2 * hidden_size * visible_size: 2 * hidden_size * visible_size +
               hidden_size].reshape(hidden_size, 1)
    b2 = theta[2 * hidden_size * visible_size + hidden_size:].\
        reshape(visible_size, 1)

    # initialize gradients
    W1_grad = np.zeros(W1.shape)
    W2_grad = np.zeros(W2.shape)
    b1_grad = np.zeros(b1.shape)
    b2_grad = np.zeros(b2.shape)

    # initialize delta items
    W1_delta = np.zeros(W1.shape)
    W2_delta = np.zeros(W2.shape)
    b1_delta = np.zeros(b1.shape)
    b2_delta = np.zeros(b2.shape)

    # initialize network layers
    z = dict()  # keys are from 2 to number of layers
    a = dict()  # keys are from 1 to number of layers
    sigma = dict()  # keys are from 2 to number of layers

    # for loop version
    '''
    # Sparsity pre-feedforward process, to get rho
    # rho is the average activation of hidden layer units
    rho = np.zeros(b1.shape)
    # paralization, to be updated
    for i in range(data.shape[1]):
        a[1] = data[:, i].reshape(visible_size, 1)
        z[2] = np.dot(W1,  a[1]) + b1
        a[2] = sigmoid(z[2])
        rho += a[2]
    rho /= data.shape[1]
    '''

    # dpark version
    # Sparsity pre-feedforward process, to get rho
    # rho is the average activation of hidden layer units
    def map_rho_iter(dat):
        a[1] = dat.reshape(visible_size, 1)
        z[2] = np.dot(W1,  a[1]) + b1
        a[2] = sigmoid(z[2])
        rho_iter = a[2].reshape(a[2].size)
        return rho_iter

    #print "compute_grad rho collecting"
    rho_collect = dpark.makeRDD(
        data.T, 300
        ).map(
        map_rho_iter
        ).collect()
    rho_collect = np.array(rho_collect).T
    rho = np.sum(rho_collect, 1).reshape(rho_collect.shape[0], 1)
    rho /= data.shape[1]
    # dpark finished

    # for loop version
    '''
    # Backpropogation
    # paralization, to be updated
    #print "for-loop compute_grad for der"
    for i in range(data.shape[1]):
        a[1] = data[:, i].reshape(visible_size, 1)
        z[2] = np.dot(W1,  a[1]) + b1
        a[2] = sigmoid(z[2])
        z[3] = np.dot(W2, a[2]) + b2
        a[3] = sigmoid(z[3])
        sigma[3] = -(a[1] - a[3]) * (a[3] * (1 - a[3]))
        sparsity_sigma = -sparsity_param / rho +\
            (1 - sparsity_param) / (1 - rho)
        sigma[2] = (np.dot(W2.T, sigma[3]) + beta * sparsity_sigma) *\
                   (a[2] * (1 - a[2]))

        W2_partial_derivative = np.dot(sigma[3], a[2].T)
        b2_partial_derivative = sigma[3]
        W1_partial_derivative = np.dot(sigma[2], a[1].T)
        b1_partial_derivative = sigma[2]

        W1_delta += W1_partial_derivative
        W2_delta += W2_partial_derivative
        b1_delta += b1_partial_derivative
        b2_delta += b2_partial_derivative

    '''

    # dpark version
    # new version
    # Backpropogation
    #print "compute_grad der new version!"
    W1_delta_acc = dpark.accumulator(np.zeros(shape=W1.shape))
    W2_delta_acc = dpark.accumulator(np.zeros(shape=W2.shape))
    b1_delta_acc = dpark.accumulator(np.zeros(shape=b1.shape))
    b2_delta_acc = dpark.accumulator(np.zeros(shape=b2.shape))

    def map_der_iter(dat):
        a[1] = dat.reshape(visible_size, 1)
        z[2] = np.dot(W1, a[1]) + b1
        a[2] = sigmoid(z[2])
        z[3] = np.dot(W2, a[2]) + b2
        a[3] = sigmoid(z[3])
        sigma[3] = -(a[1] - a[3]) * (a[3] * (1 - a[3]))
        sparsity_sigma = -sparsity_param / rho +\
            (1 - sparsity_param) / (1 - rho)
        sigma[2] = (np.dot(W2.T, sigma[3]) + beta * sparsity_sigma) *\
                   (a[2] * (1 - a[2]))

        # accumulators
        W2_delta_acc.add(np.dot(sigma[3], a[2].T))
        W1_delta_acc.add(np.dot(sigma[2], a[1].T))
        b2_delta_acc.add(sigma[3])
        b1_delta_acc.add(sigma[2])

    dpark.makeRDD(
        data.T, 300
        ).map(
        map_der_iter
        ).collect()

    W1_delta = W1_delta_acc.value
    W2_delta = W2_delta_acc.value
    b1_delta = b1_delta_acc.value
    b2_delta = b2_delta_acc.value
    # dpark finished

    # gradient computing
    W1_grad = W1_delta / data.shape[1] + lamb * W1
    W2_grad = W2_delta / data.shape[1] + lamb * W2
    b1_grad = b1_delta / data.shape[1]
    b2_grad = b2_delta / data.shape[1]

    # return vector version 'grad'
    grad = np.hstack(([], W1_grad.reshape(W1.size), W2_grad.reshape(W2.size),
                      b1_grad.reshape(b1.size), b2_grad.reshape(b2.size)))

    return grad
