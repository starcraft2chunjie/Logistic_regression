import numpy as np
import scipy as sp
def loadDataSet():
    dataIn = []
    labelIn = []
    fr = open('testSet.txt')
    for line in fr.readline():
        lineArr = line.strip().split()
        dataIn.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelIn.append([int(lineArr[2])])
    return dataIn, labelIn

def sigmoid(x):
    x = np.exp(-1 * x)
    x = 1 / (1 + x)
    return x

def costFunction_gradient(dataMatrix, labelMatrix, theta, length):
    trainData = sigmoid(dataMatrix * theta)
    costfunction = (labelMatrix * np.log(trainData) * -1 - (1 - labelMatrix) * np.log(1 - trainData)) / length
    gradient = dataMatrix.transpose() * (trainData - labelMatrix) / length
    alpha = 0.01
    theta = theta - alpha * gradient
    return theta, costfunction

def test(dataIn, labelIn):
    dataMatrix = np.mat(dataIn)
    labelMatrix = np.mat(labelIn)
    m, n = np.shape(dataMatrix)
    theta = np.ones(n, 1)
    maxcycle = 500
    for i in range(maxcycle):
        theta, costfunction = costFunction_gradient(dataMatrix, labelMatrix, theta, m)
        print(costfunction)
    print(theta)










