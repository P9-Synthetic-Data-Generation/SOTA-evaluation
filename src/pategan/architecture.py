import tensorflow as tf


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.0)
    return tf.compat.v1.random_normal(shape=size, stddev=xavier_stddev)


def generator(z, n_columns):

    generator_h_dim = int(4 * n_columns)

    G_W1 = tf.Variable(xavier_init([n_columns, generator_h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[generator_h_dim]))

    G_W2 = tf.Variable(xavier_init([generator_h_dim, generator_h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[generator_h_dim]))

    G_W3 = tf.Variable(xavier_init([generator_h_dim, n_columns]))
    G_b3 = tf.Variable(tf.zeros(shape=[n_columns]))

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    G_h1 = tf.nn.tanh(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
    G_out = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)

    return G_out, theta_G


def student(x, n_columns):
    S_W1 = tf.Variable(xavier_init([n_columns, n_columns]))
    S_b1 = tf.Variable(tf.zeros(shape=[n_columns]))

    S_W2 = tf.Variable(xavier_init([n_columns, 1]))
    S_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_S = [S_W1, S_W2, S_b1, S_b2]

    S_h1 = tf.nn.relu(tf.matmul(x, S_W1) + S_b1)
    S_out = tf.matmul(S_h1, S_W2) + S_b2

    return S_out, theta_S
