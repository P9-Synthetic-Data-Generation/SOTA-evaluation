import keras


def build_generator(noise_dim):
    model = keras.Sequential(
        [
            keras.layers.Dense(128, activation="relu", input_dim=noise_dim),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(46, activation="sigmoid"),
        ]
    )
    return model


def build_teacher_discriminators(num_teachers):
    return [
        keras.Sequential(
            [
                keras.layers.Dense(256, activation="relu", input_dim=46),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        for _ in range(num_teachers)
    ]


def build_student_discriminator():
    model = keras.Sequential(
        [
            keras.layers.Dense(256, activation="relu", input_dim=46),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model
