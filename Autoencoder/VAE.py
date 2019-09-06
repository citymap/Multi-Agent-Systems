from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pygame
import math
import time

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test = data

    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)

    # display a 30x30 2D manifold of digits
    z_sample = np.array([[0, 0]])
    x_decoded = decoder.predict(z_sample)
    digit = x_decoded[0].reshape(28, 28)

# Generate data

display = pygame.display.set_mode((350, 350))


# Convert coordinates form cartesian to screen coordinates (used to draw in pygame screen)
def cartesian_to_screen(car_pos):
    factor = 1
    screen_pos = np.array([center[0] * factor + car_pos[0], center[1] * factor - car_pos[1]]) / factor
    screen_pos = screen_pos.astype(int)
    return screen_pos

def screen_to_cartesian(screen_pos):
    factor = 1
    car_pos = np.array([screen_pos[0] - center[0], center[1] - screen_pos[1]]) * factor
    car_pos = car_pos.astype(float)
    return car_pos


def draw(img):
    pygame.event.get()
    screen.fill((0, 0, 0))
    for i in range(0, 28):
        for j in range(0, 28):
            color = int(img[j,i]*255)
            pixel = pygame.Rect(i*width/28,j*width/28, width/28, width/28)
            pygame.draw.rect(screen, (color,color,color), pixel)
    # for pt in x_encoded:
    #     pygame.draw.circle(screen, (255, 0, 0), cartesian_to_screen(pt*60), 3)

    pygame.display.flip()

# Screen parameters
width = 28
height = 28
center = np.array([width/2, height/2])
screen = pygame.display.set_mode((width, height))

# Colors
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)
yellow = (255,255, 0)

n_samples = 25000
angles = np.random.randn(n_samples)*3
positions = np.random.randn(n_samples)*3
# angles = np.where(angles < -math.pi, -math.pi, angles)
# angles = np.where(angles > math.pi, math.pi,angles)
#
# positions = np.where(positions < -5, -5, positions)
# positions = np.where(positions >5, 5, positions)

print(positions)

data = np.zeros((n_samples,28,28))

for i in range(n_samples):
    pygame.event.get()

    screen.fill((0, 0, 0))
    # p11 = np.array([positions[i], -14])
    # p12 = p11 + 28*np.array([math.sin(angles[i]),1])
    # p21 = np.array([positions[i] + 5, -14])
    # p22 = p21 + 28 * np.array([math.sin(angles[i]), 1])
    # pygame.draw.line(screen, white, cartesian_to_screen(p11),
    #                  cartesian_to_screen(p12), 3)
    pygame.draw.circle(screen, white, cartesian_to_screen(np.array([angles[i],positions[i]])), 3)

    # pygame.draw.line(screen, white, cartesian_to_screen(p21),
    #                  cartesian_to_screen(p22), 3)
    # pygame.display.flip()

    display.blit(screen, (0, 0))
    pygame.display.update()

    # Convert the window in black color(2D) into a matrix
    screen_px = pygame.surfarray.array2d(display)
    screen_px = screen_px/ np.max(screen_px)

    screen_px = np.flip(np.rot90(np.rot90(np.rot90(screen_px))),axis=1)
    data[i] = screen_px

x_train = data[:int(9/10*n_samples),:,:].astype('float32')
x_test = data[int(9/10*n_samples):,:,:].astype('float32')
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# # MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
image_size = 28
original_dim = image_size * image_size
# x_train = np.reshape(x_train, [-1, original_dim])
# x_test = np.reshape(x_test, [-1, original_dim])
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 3
epochs = 5

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = x_test

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    # plot_model(vae,
    #            to_file='vae_mlp.png',
    #            show_shapes=True)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_mlp_mnist.h5')
    x_encoded = encoder.predict(x_test,batch_size=batch_size)
    x_decoded = decoder.predict(x_encoded[2])
    digit = x_decoded[0].reshape(28, 28)
    draw(digit)
    time.sleep(2)

    x_mean = np.mean(x_encoded[2], axis=0)
    x_stds = np.std(x_encoded[2], axis=0)
    x_cov = np.cov((x_encoded[2] - x_mean).T)
    e, v = np.linalg.eig(x_cov)
    e_list = e.tolist()
    e_list.sort(reverse=True)

    print(x_encoded[2][0])
    plt.figure(figsize=(6, 6))
    plt.scatter(x_encoded[2][:, 0], x_encoded[2][:, 1], c=angles[int(9 / 10 * n_samples):])

    plt.colorbar()
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.scatter(x_encoded[2][:, 0], x_encoded[2][:, 1], c=positions[int(9 / 10 * n_samples):])

    plt.colorbar()
    plt.show()
    z_sample = x_mean + np.dot(v, (x_encoded[2] * e).T).T
    plt.figure(figsize=(6, 6))
    plt.scatter(z_sample[:, 0], z_sample[:, 1])
    plt.show()
    # Screen parameters
    width = 600
    height = 600
    center = np.array([width / 2, height / 2])
    screen = pygame.display.set_mode((width, height))
    sliders = np.zeros(latent_dim)
    k = 0
    state = 'still'
    rand_vecs = np.zeros(latent_dim)
    while True:
        if state == 'up':
            sliders[k] += 0.04
        if state == 'down':
            sliders[k] -= 0.04

        for event in pygame.event.get():
            # When click event
            if event.type == pygame.KEYDOWN:
                print(event.key)
                if event.key == pygame.K_LEFT:
                    k -= 1
                if event.key == pygame.K_RIGHT:
                    k += 1
                if event.key == pygame.K_UP:
                    state = 'up'
                if event.key == pygame.K_DOWN:
                    state = 'down'
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    state = 'still'
                if event.key == pygame.K_DOWN:
                    state = 'still'
            if event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos

                cartesian_pos = screen_to_cartesian(mouse_pos)
                rand_vecs = cartesian_pos / 300
                rand_vecs = np.array(rand_vecs)
                rand_vecs = list(rand_vecs)
                for i in range(2, latent_dim):
                    rand_vecs.append(0)
                rand_vecs = np.array(rand_vecs)
                print(rand_vecs)

        # draw(x_test[0].reshape(28,28))
        # time.sleep(1)
        # z_sample = np.array([sliders])
        # z_sample = encoder.predict(x_test[0].reshape(1,784))[0]
        # z_sample = x_mean + np.dot(v, (rand_vecs * e).T).T
        z_sample = rand_vecs
        z_sample = x_mean + np.dot(v, (rand_vecs * e).T).T

        print(z_sample, z_sample.shape)

        # z_sample = np.array([z_sample[0],z_sample[1],0,0,0])
        x_decoded = decoder.predict(np.array([z_sample]))
        digit = x_decoded[0].reshape(28, 28)
        draw(digit)
        # time.sleep(1)
