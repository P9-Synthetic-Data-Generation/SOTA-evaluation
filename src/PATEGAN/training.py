import os
import sys

import keras
import numpy as np
import tensorflow as tf
from architecture import build_generator, build_student_discriminator, build_teacher_discriminators

sys.path.append("src")
from utils.data_handling import data_loader, save_synthetic_data


def pate_aggregate(teacher_votes, dp_noise):
    """
    Apply the PATE mechanism to aggregate teacher votes with differential privacy.
    """
    counts = np.sum(teacher_votes, axis=0)
    noise = np.random.laplace(0, dp_noise, size=counts.shape)
    noisy_counts = counts + noise
    return (noisy_counts > 0.5).astype(float)  # Binary decision based on majority vote


def training(
    features, labels, generator, teacher_discriminators, student_discriminator, epochs, batch_size, noise_dim, dp_noise
):
    # Compile the Student Discriminator
    student_discriminator.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy"
    )

    # Compile Teacher Discriminators
    for teacher in teacher_discriminators:
        teacher.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")

    # Optimizer for the Generator
    optimizer_gen = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    loss_fn = keras.losses.BinaryCrossentropy()

    # Real data
    real_data = np.hstack([features, labels])

    for epoch in range(epochs):
        # Train Teacher Discriminators
        for teacher in teacher_discriminators:
            teacher.fit(real_data, np.ones((real_data.shape[0], 1)), epochs=1, verbose=0)

        # Generate synthetic samples
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        synthetic_samples = generator.predict(noise)

        # Teachers vote on synthetic samples
        teacher_votes = [teacher.predict(synthetic_samples) > 0.5 for teacher in teacher_discriminators]
        aggregated_labels = pate_aggregate(np.array(teacher_votes), dp_noise)  # Shape: (batch_size, 1)

        # Fix concatenation of features and aggregated labels
        synthetic_labeled = np.hstack([synthetic_samples[:, :-1], aggregated_labels])

        # Train Student Discriminator on teacher-labeled synthetic data
        student_discriminator.train_on_batch(synthetic_labeled, aggregated_labels)

        # Train Generator to fool the student discriminator
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        misleading_targets = np.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            generated_samples = generator(noise)
            student_predictions = student_discriminator(generated_samples)
            gen_loss = loss_fn(misleading_targets, student_predictions)
        gradients = tape.gradient(gen_loss, generator.trainable_variables)
        optimizer_gen.apply_gradients(zip(gradients, generator.trainable_variables))

        # Logging progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs} | Gen Loss: {gen_loss:.4f}")


if __name__ == "__main__":
    num_teachers = 10
    noise_dim = 100
    dp_noise = 0.5

    features, labels = data_loader(
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "train_data.pkl"),
        os.path.join("data", "mimic-iii_preprocessed", "pickle_data", "train_labels.pkl"),
    )

    generator = build_generator(noise_dim)
    teacher_discriminators = build_teacher_discriminators(num_teachers)
    student_discriminator = build_student_discriminator()

    training(
        features,
        labels,
        generator,
        teacher_discriminators,
        student_discriminator,
        epochs=100,
        batch_size=64,
        noise_dim=noise_dim,
        dp_noise=dp_noise,
    )

    # Generate synthetic labeled data
    noise = np.random.normal(0, 1, (500, noise_dim))
    synthetic_data = generator.predict(noise)
    save_synthetic_data(synthetic_data, "pategan_test_500_rows.csv")
