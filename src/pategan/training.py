import os
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from architecture import generator, student


def pate_lambda(x, teacher_models, lamda):
    """Returns PATE_lambda(x).

    Args:
      - x: feature vector
      - teacher_models: a list of teacher models
      - lamda: parameter

    Returns:
      - n0, n1: the number of label 0 and 1, respectively
      - out: label after adding laplace noise.
    """

    y_hat = list()

    for teacher in teacher_models:
        temp_y = teacher.predict(np.reshape(x, [1, -1]))
        y_hat = y_hat + [temp_y]

    y_hat = np.asarray(y_hat)
    n0 = sum(y_hat == 0)
    n1 = sum(y_hat == 1)

    lap_noise = np.random.laplace(loc=0.0, scale=lamda)

    out = (n1 + lap_noise) / float(n0 + n1)
    out = int(out > 0.5)

    return n0, n1, out


def training(x_partition, parameters):

    def _sample_Z(m, n):
        return np.random.uniform(-1.0, 1.0, size=[m, n])

    tf.compat.v1.reset_default_graph()

    Z = tf.compat.v1.placeholder(tf.float32, shape=[None, parameters["n_columns"]])
    Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

    # alpha initialize
    L = 20
    alpha = np.zeros([L])
    # initialize epsilon_hat
    epsilon_hat = 0

    ## Loss
    G_sample, theta_G = generator(Z)
    S_fake, theta_S = student(G_sample)

    S_loss = tf.reduce_mean(Y * S_fake) - tf.reduce_mean((1 - Y) * S_fake)
    G_loss = -tf.reduce_mean(S_fake)

    # Optimizer
    S_solver = tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-S_loss, var_list=theta_S)
    G_solver = tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=theta_G)

    clip_S = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_S]

    ## Sessions
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    ## Iterations
    while epsilon_hat < parameters["epsilon"]:

        # 1. Train teacher models
        teacher_models = list()

        for _ in range(parameters["k"]):

            Z_mb = _sample_Z(n_partition_size, n_columns)
            G_mb = sess.run(G_sample, feed_dict={Z: Z_mb})

            temp_x = x_partition[i]
            idx = np.random.permutation(len(temp_x[:, 0]))
            X_mb = temp_x[idx[:n_partition_size], :]

            X_comb = np.concatenate((X_mb, G_mb), axis=0)
            Y_comb = np.concatenate(
                (
                    np.ones(
                        [
                            n_partition_size,
                        ]
                    ),
                    np.zeros(
                        [
                            n_partition_size,
                        ]
                    ),
                ),
                axis=0,
            )

            model = LogisticRegression()
            model.fit(X_comb, Y_comb)
            teacher_models = teacher_models + [model]

        # 2. Student training
        for _ in range(parameters["n_s"]):

            Z_mb = _sample_Z(parameters["batch_size"], n_columns)
            G_mb = sess.run(G_sample, feed_dict={Z: Z_mb})
            Y_mb = list()

            for j in range(parameters["batch_size"]):
                n0, n1, r_j = pate_lambda(G_mb[j, :], teacher_models, parameters["lambda"])
                Y_mb = Y_mb + [r_j]

                # Update moments accountant
                q = np.log(2 + parameters["lambda"] * abs(n0 - n1)) - np.log(4.0) - (parameters["lambda"] * abs(n0 - n1))
                q = np.exp(q)

                # Compute alpha
                for l in range(L):
                    temp1 = 2 * (parameters["lambda"] ** 2) * (l + 1) * (l + 2)
                    temp2 = (1 - q) * (((1 - q) / (1 - q * np.exp(2 * parameters["lambda"]))) ** (l + 1)) + q * np.exp(
                        2 * parameters["lambda"] * (l + 1)
                    )

                    alpha[l] = alpha[l] + np.min([temp1, np.log(temp2[0])])

            # PATE labels for G_mb
            Y_mb = np.reshape(np.asarray(Y_mb), [-1, 1])

            # Update student
            _, D_loss_curr, _ = sess.run([S_solver, S_loss, clip_S], feed_dict={Z: Z_mb, Y: Y_mb})

        # Generator Update
        Z_mb = _sample_Z(parameters["batch_size"], n_columns)
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_mb})

        # epsilon_hat computation
        curr_list = list()
        for l in range(L):
            temp_alpha = (alpha[l] + np.log(1 / parameters["delta"])) / float(l + 1)
            curr_list = curr_list + [temp_alpha]

        epsilon_hat = np.min(curr_list)

    ## Outputs
    x_train_hat = sess.run([G_sample], feed_dict={Z: _sample_Z(parameters["n_rows"], n_columns)})[0]

    return x_train_hat


if __name__ == "__main__":
    data = np.load(os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "data.pkl"), allow_pickle=True)
    labels = np.load(os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "labels.pkl"), allow_pickle=True)
    reshaped_data = data.reshape(8228, -1)
    data = np.hstack((reshaped_data, labels.reshape(-1, 1)))

    train_data, test_data = train_test_split(data, train_size=0.5)

    n_rows, n_columns = train_data.shape
    student_h_dim = int(n_columns)
    generator_h_dim = int(4 * n_columns)

    parameters = {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "n_s": 1,
        "batch_size": 64,
        "k": 10,
        "epsilon": 1.0,
        "delta": 0.00001,
        "lambda": 1.0,
    }

    x_partition = list()
    n_partition_size = int(n_rows / parameters["k"])

    idx = np.random.permutation(n_rows)

    for i in range(parameters["k"]):
        temp_idx = idx[int(i * n_partition_size) : int((i + 1) * n_partition_size)]
        temp_x = data[temp_idx, :]
        x_partition = x_partition + [temp_x]

    synth_data = training(x_partition, parameters)
