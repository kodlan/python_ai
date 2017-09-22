import tensorflow as tf
from catdata import generate_minibatch, IMAGE_SIZE
import matplotlib.pyplot as plt
import numpy as np

def create_placeholders(n_x, n_y):
    X_p = tf.placeholder(tf.float32, shape=(n_x, None), name="X")
    Y_p = tf.placeholder(tf.float32, shape=(1, n_y), name="Y")
    return X_p, Y_p


def init_params():
    W1 = tf.get_variable("W1", [25, IMAGE_SIZE], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable("W3", [1, 12], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [1, 1], initializer=tf.contrib.layers.xavier_initializer())

    return {"W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "W3": W3,
            "b3": b3}


def forward_prop(X, params):
    Z1 = tf.add(tf.matmul(params["W1"], X), params["b1"])
    A1 = tf.nn.relu(Z1)

    Z2 = tf.add(tf.matmul(params["W2"], A1), params["b2"])
    A2 = tf.nn.relu(Z2)

    Z3 = tf.add(tf.matmul(params["W3"], A2), params["b3"])

    return Z3


def calc_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def plot_cost(costs):
    plt.plot(np.squeeze(costs))
    plt.xlabel('iterations')
    plt.ylabel("cost")
    plt.show()


def test(X_p, Y_predict, sess):
    (files, X, Y) = generate_minibatch(10)
    result = sess.run(Y_predict, feed_dict={X_p: X})
    print ("result = ")
    print (result)
    print (result > 0.5)
    print
    print ("actual results = ")
    print (Y)
    print

    for i in range(Y.shape[1]):
        if (result[0][i] > 0.5) != bool(Y[0][i]):
            print ("miss qualified image : %s" % files[i])


def model(learning_rate=0.001, num_epoch=1500, number_of_images=320, minibatch_size=32):
    costs = []

    n_x = IMAGE_SIZE
    n_y = minibatch_size

    tf.set_random_seed(1)
    tf.reset_default_graph()

    X_p, Y_p = create_placeholders(n_x, n_y)

    params = init_params()
    Z3 = forward_prop(X_p, params)
    cost = calc_cost(Z3, Y_p)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    Y_predict = tf.sigmoid(Z3)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epoch):
            epoch_cost = 0
            num_minibatches = int(number_of_images / minibatch_size)

            for mi in range(num_minibatches):
                (_, X, Y) = generate_minibatch(minibatch_size)

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X_p: X, Y_p: Y})

                epoch_cost += minibatch_cost / num_minibatches

            if epoch % 100 == 0:
                print ("Cost for epoch %i: %f" % (epoch, epoch_cost))

            costs.append(epoch_cost)

        test(X_p, Y_predict, sess)

    return costs


costs = model()
plot_cost(costs)
