import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def logistic_reg(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.001):
    w, b = init_params(X_train.shape[0])

    w, b, dw, db, cost_list = gradient_descent(w, b, X_train, Y_train, num_iterations, learning_rate)

    Y_prediction_test = compute(w, b, X_test)
    Y_prediction_train = compute(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    plot_cost(cost_list, learning_rate)


def plot_cost(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def gradient_descent(w, b, X, Y, num_iterations, learning_rate):
    cost_list = []

    for i in range(num_iterations):
        dw, db, cost = propagate(X, Y, w, b)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            cost_list.append(cost)
            print ("Cost value after %i iteration: %f" %(i, cost))

    return w, b, dw, db, cost_list


def init_params(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def propagate(X, Y, w, b):
    m = X.shape[1]

    # forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = calc_cost(m, A, Y)
    cost = np.squeeze(cost)

    # backward propagation
    dw = (1. / m) * np.dot(X, (A - Y).T)
    db = 1. / m * np.sum (A - Y)

    return dw, db, cost


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def calc_cost(m, A, Y):
    return (-1./m) * np.sum(Y * np.log(A) + ((1. - Y) * np.log(1. - A)), keepdims=True)


def compute(w, b, X):
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    Y_prediction = A

    return Y_prediction


df = pd.read_csv("train.csv")

X_train = df[df.columns[0]]
X_train = np.array(X_train)

Y_train = df[df.columns[1]]
Y_train = np.array(Y_train)

m = X_train.shape[0]
X_train = X_train.reshape(m, 1)
Y_train = Y_train.reshape(m, 1)
X_train = X_train.T
Y_train = Y_train.T


df = pd.read_csv("test.csv")

X_test = df[df.columns[0]]
X_test = np.array(X_test)

Y_test = df[df.columns[1]]
Y_test = np.array(Y_test)

m = X_test.shape[0]
X_test = X_test.reshape(m, 1)
Y_test = Y_test.reshape(m, 1)
X_test = X_test.T
Y_test = Y_test.T

print (X_train.shape)
print (Y_train.shape)

print (X_test.shape)
print (Y_test.shape)

logistic_reg(X_train, Y_train, X_test, Y_test)