from __future__ import print_function

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import argparse
import random as rn
from collections import defaultdict
import pickle

import numpy as np
import tensorflow as tf
from keras import backend, layers, models, utils, optimizers
from privacy_accountant import accountant
from custom_keras.noisy_optimizers import NoisyAdam

# Constants
TRAINING_SIZE = 7500
TARGET_EPS = [0.125, 0.25, 0.5, 1, 2, 4, 8]
LATENT_SIZE = 100

# Set image data format
backend.set_image_data_format("channels_first")
priv_accountant = accountant.GaussianMomentsAccountant(TRAINING_SIZE)


def build_generator(latent_size):
    """Builds the generator model."""
    print("Building Generator...")
    cnn = models.Sequential([
        layers.Dense(256, input_dim=latent_size, activation="relu"),
        layers.Dense(32 * 9 * 5, activation="relu"),
        layers.Reshape((32, 9, 5)),
        layers.UpSampling2D(size=(6, 6)),
        layers.Conv2D(256, 5, padding="same", activation="relu", kernel_initializer="glorot_normal"),
        layers.Conv2D(1, 4, strides=6, padding="same", activation="linear", kernel_initializer="glorot_normal")
    ])
    cnn.summary()

    latent = layers.Input(shape=(latent_size,))
    patient_class = layers.Input(shape=(1,), dtype="int32")
    cls = layers.Flatten()(layers.Embedding(2, latent_size, embeddings_initializer="glorot_normal")(patient_class))
    h = layers.multiply([latent, cls])
    fake_patient = cnn(h)

    return models.Model([latent, patient_class], fake_patient)


def build_discriminator():
    """Builds the discriminator model."""
    print("Building Discriminator...")
    cnn = models.Sequential([
        layers.Conv2D(32, 3, padding="same", strides=2, input_shape=(1, 9, 5)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(64, 3, padding="same", strides=1),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1024, activation="relu")
    ])
    cnn.summary()

    patient = layers.Input(shape=(1, 9, 5))
    features = cnn(patient)
    fake = layers.Dense(1, activation="sigmoid", name="generation")(features)
    aux = layers.Dense(2, activation="softmax", name="auxiliary")(features)

    return models.Model(patient, [fake, aux])


def train_acgan(args, generator, discriminator, combined, X_train, y_train, X_test, y_test):
    """Train the ACGAN model."""
    num_train, num_test = X_train.shape[0], X_test.shape[0]
    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        num_batches = TRAINING_SIZE
        progress_bar = utils.Progbar(target=num_batches)

        random_indices = np.random.randint(0, TRAINING_SIZE, size=TRAINING_SIZE)
        epoch_gen_loss, epoch_disc_loss = [], []

        for index in range(num_batches):
            progress_bar.update(index)

            # Generate noise and labels
            noise = np.random.uniform(-1, 1, (args.batch_size, LATENT_SIZE))
            sampled_labels = np.random.randint(0, 2, args.batch_size)

            # Get real samples
            real_images = np.expand_dims(X_train[random_indices[index]], axis=0)
            real_labels = np.expand_dims(y_train[random_indices[index]], axis=0)

            # Generate fake samples
            fake_images = generator.predict([noise, sampled_labels.reshape((-1, 1))])

            # Combine real and fake samples
            X = np.concatenate((real_images, fake_images))
            y = np.array([1] * args.batch_size + [0] * args.batch_size)
            aux_y = np.concatenate((real_labels, sampled_labels), axis=0)

            # Train discriminator
            epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))

            # Train generator
            noise = np.random.uniform(-1, 1, (2 * args.batch_size, LATENT_SIZE))
            sampled_labels = np.random.randint(0, 2, 2 * args.batch_size)
            trick = np.ones(2 * args.batch_size)

            epoch_gen_loss.append(
                combined.train_on_batch(
                    [noise, sampled_labels.reshape((-1, 1))], 
                    [trick, sampled_labels]
                )
            )

        # Evaluate models
        print("\nTesting at Epoch {}/{}:".format(epoch + 1, args.epochs))
        noise = np.random.uniform(-1, 1, (num_test, LATENT_SIZE))
        sampled_labels = np.random.randint(0, 2, num_test)
        fake_images = generator.predict([noise, sampled_labels.reshape((-1, 1))])
        
        trick = np.ones(num_test)

        X_test_combined = np.concatenate((X_test, fake_images))
        y_test_combined = np.array([1] * num_test + [0] * num_test)
        aux_test_combined = np.concatenate((y_test, sampled_labels), axis=0)

        disc_test_loss = discriminator.evaluate(X_test_combined, [y_test_combined, aux_test_combined], verbose=0)
        #gen_test_loss = combined.evaluate([noise, sampled_labels.reshape((-1, 1))], [np.ones(2 * num_test), sampled_labels], verbose=0)
        gen_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))], 
            [trick, sampled_labels], 
            verbose=0
        )


        train_history['generator'].append(np.mean(epoch_gen_loss, axis=0))
        train_history['discriminator'].append(np.mean(epoch_disc_loss, axis=0))
        test_history['generator'].append(gen_test_loss)
        test_history['discriminator'].append(disc_test_loss)

        print(f"Epoch {epoch + 1} Summary:")
        print(f"Generator Train Loss: {train_history['generator'][-1]}")
        print(f"Discriminator Train Loss: {train_history['discriminator'][-1]}")
        print(f"Generator Test Loss: {test_history['generator'][-1]}")
        print(f"Discriminator Test Loss: {test_history['discriminator'][-1]}")

        generator.save(f"{args.output_dir}/generator_epoch_{epoch + 1}.h5")
        discriminator.save(f"{args.output_dir}/discriminator_epoch_{epoch + 1}.h5")

    return train_history, test_history


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data_path = "data/mimic-iii_preprocessed/pickle_data"
    X_input = pickle.load(open(os.path.join(data_path, "data.pkl"), "rb"))
    y_input = pickle.load(open(os.path.join(data_path, "labels.pkl"), "rb"))

    X_train, X_test = X_input[:TRAINING_SIZE], X_input[TRAINING_SIZE:]
    y_train, y_test = y_input[:TRAINING_SIZE], y_input[TRAINING_SIZE:]
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    # Build models
    generator = build_generator(LATENT_SIZE)
    discriminator = build_discriminator()
    discriminator.compile(optimizer=optimizers.Adam(learning_rate=args.lr, beta_1=0.5),
                          loss=["binary_crossentropy", "sparse_categorical_crossentropy"])

    latent = layers.Input(shape=(LATENT_SIZE,))
    image_class = layers.Input(shape=(1,), dtype="int32")
    fake_image = generator([latent, image_class])
    discriminator.trainable = False
    fake, aux = discriminator(fake_image)
    combined = models.Model([latent, image_class], [fake, aux])
    combined.compile(optimizer=optimizers.Adam(learning_rate=args.lr, beta_1=0.5),
                     loss=["binary_crossentropy", "sparse_categorical_crossentropy"])

    # Train models
    train_acgan(args, generator, discriminator, combined, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
