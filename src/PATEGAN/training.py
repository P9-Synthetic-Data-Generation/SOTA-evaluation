import tensorflow as tf
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Constants
NUM_FEATURES = 45  # Time-series features
LABEL_DIM = 1  # Binary label
NOISE_DIM = 100  # Dimension of noise input
NUM_TEACHERS = 5  # Number of teacher discriminators
BATCH_SIZE = 64
EPOCHS = 100
DIFFERENTIAL_PRIVACY_NOISE = 0.5  # Laplace noise scale for PATE mechanism


# -----------------------------
# Data Preprocessing Function
# -----------------------------
def preprocess_data(data):
    """
    Preprocess medical time-series data.
    Assumes data is a pandas DataFrame with features and binary labels.
    """
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values.reshape(-1, 1)
    # Normalize features to range [0, 1]
    features = (features - features.min()) / (features.max() - features.min())
    return features, labels


# -----------------------------
# Generator Model
# -----------------------------
def build_generator():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(128, activation="relu", input_dim=NOISE_DIM),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(NUM_FEATURES + LABEL_DIM, activation="sigmoid"),  # Output: features + label
        ]
    )
    return model


# -----------------------------
# Teacher Discriminators
# -----------------------------
def build_teacher_discriminators():
    return [
        tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu", input_dim=NUM_FEATURES + LABEL_DIM),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),  # Binary classification
            ]
        )
        for _ in range(NUM_TEACHERS)
    ]


# -----------------------------
# Student Discriminator
# -----------------------------
def build_student_discriminator():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(256, activation="relu", input_dim=NUM_FEATURES + LABEL_DIM),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


# -----------------------------
# PATE Mechanism
# -----------------------------
def pate_aggregate(teacher_votes):
    """
    Apply the PATE mechanism to aggregate teacher votes with differential privacy.
    """
    counts = np.sum(teacher_votes, axis=0)
    noise = np.random.laplace(0, DIFFERENTIAL_PRIVACY_NOISE, size=counts.shape)
    noisy_counts = counts + noise
    return (noisy_counts > 0.5).astype(float)  # Binary decision based on majority vote


# -----------------------------
# Training Loop
# -----------------------------
def train_pate_gan(generator, teacher_discriminators, student_discriminator, features, labels):
    # Compile the Student Discriminator
    student_discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")

    # Compile Teacher Discriminators
    for teacher in teacher_discriminators:
        teacher.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")

    # Optimizer for the Generator
    optimizer_gen = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Real data
    real_data = np.hstack([features, labels])

    for epoch in range(EPOCHS):
        # Train Teacher Discriminators
        for teacher in teacher_discriminators:
            teacher.fit(real_data, np.ones((real_data.shape[0], 1)), epochs=1, verbose=0)

        # Generate synthetic samples
        noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
        synthetic_samples = generator.predict(noise)

        # Teachers vote on synthetic samples
        teacher_votes = [teacher.predict(synthetic_samples) > 0.5 for teacher in teacher_discriminators]
        aggregated_labels = pate_aggregate(np.array(teacher_votes))  # Shape: (BATCH_SIZE, 1)

        # Fix concatenation of features and aggregated labels
        synthetic_labeled = np.hstack([synthetic_samples[:, :-1], aggregated_labels])

        # Train Student Discriminator on teacher-labeled synthetic data
        student_discriminator.train_on_batch(synthetic_labeled, aggregated_labels)

        # Train Generator to fool the student discriminator
        noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
        misleading_targets = np.ones((BATCH_SIZE, 1))
        with tf.GradientTape() as tape:
            generated_samples = generator(noise)
            student_predictions = student_discriminator(generated_samples)
            gen_loss = loss_fn(misleading_targets, student_predictions)
        gradients = tape.gradient(gen_loss, generator.trainable_variables)
        optimizer_gen.apply_gradients(zip(gradients, generator.trainable_variables))

        # Logging progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Gen Loss: {gen_loss:.4f}")


# -----------------------------
# Main Code
# -----------------------------
# Load your medical dataset as a pandas DataFrame
# Replace `load_your_dataset()` with your actual data loading function
# import pandas as pd
# data = load_your_dataset()
# features, labels = preprocess_data(data)

# Simulated data for demonstration
features = np.random.rand(1000, NUM_FEATURES)
labels = np.random.randint(0, 2, size=(1000, 1))

# Build models
generator = build_generator()
teacher_discriminators = build_teacher_discriminators()
student_discriminator = build_student_discriminator()

# Train PATE-GAN
train_pate_gan(generator, teacher_discriminators, student_discriminator, features, labels)

# Generate synthetic labeled data
noise = np.random.normal(0, 1, (10, NOISE_DIM))
synthetic_data = generator.predict(noise)
print("Synthetic data (features + labels):\n", synthetic_data)
