import pandas as pd
import numpy as np
import tensorflow as tf

# def tf_log(X, Y):
def tf_log():
    x = tf.placeholder(tf.float32, [None, 1])

    W = tf.Variable(tf.zeros([1,1]))
    b = tf.Variable(tf.zeros([1]))

    y = tf.matmul(x, W) + b

    yhat = tf.placeholder(tf.float32, [None, 1])

    cost = tf.reduce_sum(tf.square(yhat - y))

    train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # for i, val in enumerate(X):
    for i in range(1,100):
        # xval = np.array([val])
        xval = np.array([[i]])
        # yval = np.array([Y[i]])
        yval = np.array([[2*i]])
        feed = {x:  xval, yhat: yval}

        _, wval, bval, costval  = sess.run([train_step, W, b, cost], feed_dict=feed)

        if i % 10 == 0:
            print("%d iteration" %i)
            print("    W = %f" % wval)
            print("    b = %f" % bval)
            print("    cost = %f" % costval)

    sess.close()

# df = pd.read_csv("train.csv")

# X_train = df[df.columns[0]]
# X_train = np.array(X_train)

# Y_train = df[df.columns[1]]
# Y_train = np.array(Y_train)

# m = X_train.shape[0]
# print ("Number of samples = %i" % m)
# X_train = X_train.reshape(m, 1)
# Y_train = Y_train.reshape(m, 1)

# tf_log(X_train, Y_train)
tf_log()