from keras.models import Model, load_model
from keras.layers import Dense, Input

from keras.datasets.mnist import load_data
import numpy as np

import matplotlib.pyplot as plt

# data
(x_train, y_train), (x_test, y_test) = load_data()
x_train, x_test = x_train / 255., x_test / 255.

width, height = x_train.shape[1], x_train.shape[2]
n_train, n_test = x_train.shape[0], x_test.shape[0]
x_train, x_test = x_train.reshape(n_train, -1), x_test.reshape(n_test, -1)

data_dim = x_train.shape[-1]


def create_autoencoder():
    # encoder
    encoder_input = Input([data_dim])
    x = Dense(30, activation='tanh')(encoder_input)
    encoder_output = Dense(2, activation='linear')(x)
    encoder = Model(encoder_input, encoder_output)

    # decoder
    decoder_input = Input(batch_shape=encoder.output_shape)
    x = Dense(30, activation='tanh')(decoder_input)
    decoder_output = Dense(data_dim, activation='linear')(x)
    decoder = Model(decoder_input, decoder_output)

    # full model
    autoencoder_output = decoder(encoder(encoder_input))
    autoencoder = Model(encoder_input, autoencoder_output)
    autoencoder.compile(optimizer='adam', loss='mse')

    return encoder, decoder, autoencoder


def load_autoencoder():
    encoder, decoder, autoencoder = create_autoencoder()

    import os.path
    if os.path.isfile('autoencoder_weights.h5'):
        autoencoder.load_weights('autoencoder_weights.h5')

    return encoder, decoder, autoencoder


def save_autoencoder(autoencoder):
    autoencoder.save_weights('autoencoder_weights.h5')


def train_model(model, x, n_epochs):
    model.fit(x, x, epochs=n_epochs, batch_size=64, verbose=2)
    save_autoencoder(autoencoder)


def plot_reconstruction(model: Model, data):
    for image in data:
        decoded = model.predict(image[None, ...])
        image_original = image.reshape(width, -1)
        image_reconstructed = decoded.reshape(width, -1)

        figure, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image_original, cmap='hot')
        ax2.imshow(image_reconstructed, cmap='magma')

        plt.show()


def plot_representation(encoder: Model, x, y):
    representation = encoder.predict(x)
    for number in range(10):
        points = representation[y == number]
        plt.scatter(points[:, 0], points[:, 1], label=number)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    encoder, decoder, autoencoder = load_autoencoder()
    train_model(autoencoder, x_train, 200)
    # plot_reconstruction(autoencoder, x_train)
    plot_representation(encoder, x_train, y_train)