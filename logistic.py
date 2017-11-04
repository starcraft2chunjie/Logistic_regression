import numpy as np
import pandas as np

def sigmoid(x, theta1, theta2):
    z = (theta1 * x + theta2).astype("float_")
    return 1.0/(1.0 + np.exp(-z))

def log_likelihood(x, y, theta1, theta2):
    sigmoid_probs = sigmoid(x, theta1, theta2)
    return np.sum(y * np.log(sigmoid_probs) + (1 - y) * (1 - sigmoid_probs))

def gradient(x, y, theta1, theta2):
    sigmoid_probs = sigmoid(x, theta1, theta2)
    return np.array([[np.sum((y - sigmoid_probs) * x), np.sum(y - sigmoid_probs)]])

def hessian(x, y, theta1, theta2):
    sigmoid_probs = sigmoid(x, theta1, theta2)
    d1 = np.sum(sigmoid_probs * (1 - sigmoid_probs) * theta1 * theta1)
    d2 = np.sum(sigmoid_probs * (1 - sigmoid_probs) * theta1)
    d3 = np.sum(sigmoid_probs * (1 - sigmoid_probs))
    H = np.array([[d1, d2], [d2, d3]])
    return H

def newtons_method(x, y):
    theta1 = 15.1
    theta2 = -.4
    lch = np.Infinity
    l = log_likelihood(x, y, theta1, theta2)
    #convergence condition
    ls = .0000000001
    max_iteration = 15
    i = 0
    while abs(l) > ls and i < max_iteration:
        i += 1
        g = gradient(x, y ,theta1, theta2)
        hess = hessian(x, y, theta1, theta2)
        H_inv = np.linalg.inv(hess)
        matrice = np.dot(H_inv, g.transpose())
        theta1dis = matrice[0][0]
        theta2dis = matrice[0][0]
        theta1 += theta1dis
        theta2 += theta2dis
        l_new = log_likelihood(x, y, theta1, theta2)
        ls = l - l_new
        l = l_new
    return np.array([theta1, theta2])

''''''