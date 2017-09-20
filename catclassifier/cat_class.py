import tensorflow as tf

n_x = 64*64*3
n_y = 1

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name="X")
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name="Y")
    return X, Y

def init_params(n_x, n_y):
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [25, n_x], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [25, n_y], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [25, n_x], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [25, n_x], initializer=tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable("W3", [25, n_x], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [25, n_x], initializer=tf.contrib.layers.xavier_initializer())

    return {"W1" : W1,
            "b1" : b1,
            "W2" : W2,
            "b2" : b2,
            "W3" : W3,
            "b3" : b3}

