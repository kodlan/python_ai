import tensorflow as tf
from catdata import generate_minibatch, IMAGE_SIZE
import matplotlib.pyplot as plt
import numpy as np


def create_placeholders(input_size, number_of_classes_to_predict):
    X_p = tf.placeholder(tf.float32, shape=(input_size, None), name="X")
    Y_p = tf.placeholder(tf.float32, shape=(number_of_classes_to_predict, None), name="Y")
    return X_p, Y_p


def init_params():
    W1 = tf.get_variable("W1", [25, IMAGE_SIZE], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable("W3", [2, 12], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [2, 1], initializer=tf.contrib.layers.xavier_initializer())

    return {"W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "W3": W3,
            "b3": b3}


def construct_network(X, params):
    Z1 = tf.add(tf.matmul(params["W1"], X), params["b1"])
    A1 = tf.nn.relu(Z1)

    Z2 = tf.add(tf.matmul(params["W2"], A1), params["b2"])
    A2 = tf.nn.relu(Z2)

    Z3 = tf.add(tf.matmul(params["W3"], A2), params["b3"])

    # output of last linear layer
    return Z3


def calc_cost(Z3, Y):
    # inputs of tf.nn.softmax_cross_entropy_with_logits are expected to be of shape (number of examples, num_classes).
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    # return tf.reduce_mean(tf.squared_difference(logits, labels))


def plot_cost(costs):
    plt.plot(np.squeeze(costs))
    plt.xlabel('iterations')
    plt.ylabel("cost")
    plt.show()


def test(X_p, Y_predict, sess):
    test_samples = 10

    (files, X, Y) = generate_minibatch(test_samples)

    result = sess.run(Y_predict, feed_dict={X_p: X})
    # result = Y_predict.eval(feed_dict={X_p: X})

    result = (tf.argmax(result, 1)).eval()
    actual_result = tf.argmax(Y, 1).eval()
    difference = tf.equal(result, actual_result).eval()

    accuracy = tf.reduce_mean(tf.cast(difference, "float")).eval()

    print ("result = ")
    print (result)
    print
    print ("actual results = ")
    print (actual_result)
    print ("accuracy = ")
    print (accuracy)

    # for i in range(test_samples):
    #     if not bool(difference[i]):
    #         print ("miss qualified image : %s" % files[i])


def model(learning_rate=0.0001, num_epoch=1500, number_of_images=128, minibatch_size=8):
    costs = []

    # tf.set_random_seed(1)
    tf.reset_default_graph()

    X_p, Y_p = create_placeholders(IMAGE_SIZE, 2)

    params = init_params()

    Z3 = construct_network(X_p, params)

    cost = calc_cost(Z3, Y_p)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    Y_predict = Z3

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
