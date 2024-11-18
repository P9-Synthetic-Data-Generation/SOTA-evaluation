from __future__ import print_function

from collections import defaultdict

import pickle
from PIL import Image

# from six.moves import range

import tensorflow as tf
import keras
import numpy as np
import random as rn
import os
import argparse

from privacy_accountant import accountant
from custom_keras.noisy_optimizers import NoisyAdam

training_size = 7500
keras.backend.set_image_data_format("channels_first")

target_eps = [0.125, 0.25, 0.5, 1, 2, 4, 8]
priv_accountant = accountant.GaussianMomentsAccountant(training_size)


def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28)
    print("Generator")
    cnn = keras.Sequential()

    cnn.add(keras.layers.Dense(256, input_dim=latent_size, activation="relu"))
    cnn.add(keras.layers.Dense(32 * 9 * 5, activation="relu"))
    cnn.add(keras.layers.Reshape((32, 9, 5)))

    # upsample to (..., 14, 14)
    cnn.add(keras.layers.UpSampling2D(size=(6, 6)))
    cnn.add(keras.layers.Conv2D(256, 5, padding="same", activation="relu", kernel_initializer="glorot_normal"))

    # take a channel axis reduction
    cnn.add(
        keras.layers.Conv2D(1, 4, strides=6, padding="same", activation="linear", kernel_initializer="glorot_normal")
    )

    # dense layer to reshape
    cnn.summary()

    # this is the z space commonly refered to in GAN papers
    latent = keras.Input(shape=(latent_size,))

    # this will be our label
    patient_class = keras.Input(shape=(1,), dtype="int32")

    # 10 classes in MNIST
    cls = keras.layers.Flatten()(
        keras.layers.Embedding(2, latent_size, embeddings_initializer="glorot_normal")(patient_class)
    )

    # hadamard product between z-space and a class conditional embedding
    h = keras.layers.multiply([latent, cls])
    fake_patient = cnn(h)

    return keras.Model([latent, patient_class], fake_patient)


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    print("Discriminator")
    cnn = keras.Sequential()
    cnn.add(keras.layers.Conv2D(32, 3, padding="same", strides=2, input_shape=(1, 9, 5)))
    cnn.add(keras.layers.LeakyReLU())
    cnn.add(keras.layers.Dropout(0.3))

    cnn.add(keras.layers.Conv2D(64, 3, padding="same", strides=1))
    cnn.add(keras.layers.LeakyReLU())
    cnn.add(keras.layers.Dropout(0.3))

    cnn.add(keras.layers.Flatten())
    cnn.add(keras.layers.Dense(1024, activation="relu"))
    cnn.add(keras.layers.Dropout(0.3))
    cnn.add(keras.layers.Dense(1024, activation="relu"))
    patient = keras.Input(shape=(1, 9, 5))

    features = cnn(patient)
    cnn.summary()

    fake = keras.layers.Dense(1, activation="sigmoid", name="generation")(features)
    # aux could probably be 1 sigmoid too...
    aux = keras.layers.Dense(2, activation="softmax", name="auxiliary")(features)

    return keras.Model(patient, [fake, aux])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--clip_value", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0002)
    # for now this should always be 1, we do not yet implement the batch and lot
    # size argument seen here - https://arxiv.org/abs/1607.00133
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--seed", type=int, default="123")
    args = parser.parse_args()

    print(args)
    epochs = args.epochs

    if args.batch_size > 1:
        raise ("Batch sizes greater than 1 are not yet implemented")

    batch_size = args.batch_size
    latent_size = 100

    # setting seed for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    rn.seed(args.seed)

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = args.lr
    adam_beta_1 = 0.5

    directory = (
        "./AC-GAN/output/"
        + str(args.prefix)
        + str(args.noise)
        + "_"
        + str(args.clip_value)
        + "_"
        + str(args.epochs)
        + "_"
        + str(args.lr)
        + "_"
        + str(args.batch_size)
        + "/"
    )

    if not os.path.exists(directory):
        os.makedirs(directory)

    if args.clip_value > 0:
        # build the discriminator
        discriminator = build_discriminator()
        discriminator.compile(
            optimizer=NoisyAdam(learning_rate=adam_lr, beta_1=adam_beta_1, clipnorm=args.clip_value, noise=args.noise),
            loss=["binary_crossentropy", "sparse_categorical_crossentropy"],
        )
    else:
        discriminator = build_discriminator()
        discriminator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
            loss=["binary_crossentropy", "sparse_categorical_crossentropy"],
        )

    # build the generator
    generator = build_generator(latent_size)
    generator.compile(
        optimizer=keras.optimizers.Adam(learning_rate=adam_lr, beta_1=adam_beta_1), loss="binary_crossentropy"
    )

    latent = keras.Input(shape=(latent_size,))
    image_class = keras.Input(shape=(1,), dtype="int32")

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = keras.Model([latent, image_class], [fake, aux])

    combined.compile(
        optimizer=keras.optimizers.Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
        loss=["binary_crossentropy", "sparse_categorical_crossentropy"],
    )

    # get our input data
    X_input = pickle.load(open(os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "data.pkl"), "rb"))
    y_input = pickle.load(open(os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "labels.pkl"), "rb"))
    print(X_input.shape, y_input.shape)

    X_train = X_input[:training_size]
    X_test = X_input[training_size:]
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    y_train = y_input[:training_size]
    y_test = y_input[training_size:]

    num_train, num_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)
    privacy_history = []

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        eps = tf.compat.v1.placeholder(tf.float32)
        delta = tf.compat.v1.placeholder(tf.float32)

        for epoch in range(epochs):
            print("Epoch {} of {}".format(epoch + 1, epochs))

            num_batches = training_size
            progress_bar = keras.utils.Progbar(target=num_batches)

            random_sample = np.random.randint(0, training_size, size=training_size)

            epoch_gen_loss = []
            epoch_disc_loss = []

            for index in range(num_batches):
                progress_bar.update(index)
                # generate a new batch of noise
                noise = np.random.uniform(-1, 1, (batch_size, latent_size))

                # get a batch of real patients
                image_batch = np.expand_dims(X_train[random_sample[index]], axis=1)
                label_batch = y_train[random_sample[index], np.newaxis]

                # sample some labels from p_c
                sampled_labels = np.random.randint(0, 2, batch_size)

                # generate a batch of fake patients, using the generated labels as a
                # conditioner. We reshape the sampled labels to be
                # (batch_size, 1) so that we can feed them into the embedding
                # layer as a length one sequence
                generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0)

                X = np.concatenate((image_batch, generated_images))
                y = np.array([1] * batch_size + [0] * batch_size)
                aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

                epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))

                # make new noise. we generate 2 * batch size here such that we have
                # the generator optimize over an identical number of patients as the
                # discriminator
                noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
                sampled_labels = np.random.randint(0, 2, 2 * batch_size)

                # we want to train the generator to trick the discriminator
                # For the generator, we want all the {fake, not-fake} labels to say
                # not-fake
                trick = np.ones(2 * batch_size)

                epoch_gen_loss.append(
                    combined.train_on_batch([noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels])
                )
            print("accum privacy, batches: " + str(num_batches))

            # separate privacy accumulation for speed
            # privacy_accum_op = priv_accountant.accumulate_privacy_spending(
            #     [None, None], args.noise, batch_size)
            # for index in range(num_batches):
            #     with tf.control_dependencies([privacy_accum_op]):
            #         spent_eps_deltas = priv_accountant.get_privacy_spent(
            #             sess, target_eps=target_eps)
            #         privacy_history.append(spent_eps_deltas)
            #     sess.run([privacy_accum_op])
            #
            # for spent_eps, spent_delta in spent_eps_deltas:
            #     print("spent privacy: eps %.4f delta %.5g" % (
            #         spent_eps, spent_delta))
            # print('priv time: ', time.clock() - priv_start_time)
            #
            # if spent_eps_deltas[-3][1] > 0.0001:
            #     raise Exception('spent privacy')

            print("\nTesting for epoch {}:".format(epoch + 1))
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (num_test, latent_size))

            # sample some labels from p_c and generate patients from them
            sampled_labels = np.random.randint(0, 2, num_test)
            generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))], verbose=False)

            print(sampled_labels[0])
            print(generated_images[0].astype(int))

            X = np.concatenate((X_test, generated_images))
            y = np.array([1] * num_test + [0] * num_test)
            aux_y = np.concatenate((y_test, sampled_labels), axis=0)

            # see if the discriminator can figure itself out...
            discriminator_test_loss = discriminator.evaluate(X, [y, aux_y], verbose=False)

            discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

            # make new noise
            noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
            sampled_labels = np.random.randint(0, 2, 2 * num_test)

            trick = np.ones(2 * num_test)

            generator_test_loss = combined.evaluate(
                [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels], verbose=False
            )

            generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

            # generate an epoch report on performance
            train_history["generator"].append(generator_train_loss)
            train_history["discriminator"].append(discriminator_train_loss)

            test_history["generator"].append(generator_test_loss)
            test_history["discriminator"].append(discriminator_test_loss)

            print("{0:<22s} | {1:4s} | {2:15s} | {3:5s}".format("component", *discriminator.metrics_names))
            print("-" * 65)

            ROW_FMT = "{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}"
            print(ROW_FMT.format("generator (train)", *train_history["generator"][-1]))
            print(ROW_FMT.format("generator (test)", *test_history["generator"][-1]))
            print(ROW_FMT.format("discriminator (train)", *train_history["discriminator"][-1]))
            print(ROW_FMT.format("discriminator (test)", *test_history["discriminator"][-1]))
            generator.save(directory + "params_generator_epoch_{0:03d}.h5".format(epoch))

            if epoch > (epochs - 10):
                discriminator.save(directory + "params_discriminator_epoch_{0:03d}.h5".format(epoch))

            pickle.dump({"train": train_history, "test": test_history}, open(directory + "acgan-history.pkl", "wb"))
            # pickle.dump({'train': train_history, 'test': test_history,
            #              'privacy': privacy_history},
            #             open(directory + 'acgan-history.pkl', 'wb'))
