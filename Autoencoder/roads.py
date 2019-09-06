# Author:   Max Martinez Ruts
# Creation: 2019

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import time
import numpy as np
import pygame
import math
import matplotlib.pyplot as plt
#dependencies (numpy, matplotlib, and keras)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import pygame

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
    pygame.display.flip()

# Screen parameters
width = 200
height = 300
center = np.array([width/2, height/2])
screen = pygame.display.set_mode((width, height))
pixel_pos = np.arange(0, width, width/28)

sliders = np.zeros(32)


#hyperparameters
batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
nb_epoch =30
epsilon_std = 1.0

#encoder
x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

print(z_mean)
print(z_log_var)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

#latent hidden state
print(z)

#decoder
# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

print(x_decoded_mean)

#loss
def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)



display = pygame.display.set_mode((350, 350))

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


angles = np.random.uniform(-math.pi,math.pi,(100000))
positions = np.random.uniform(-10,10,(100000))

data = np.zeros((100000,28,28))

for i in range(len(angles)):
    pygame.event.get()

    screen.fill((0, 0, 0))
    p11 = np.array([positions[i]-5, -14])
    p12 = p11 + 28*np.array([math.sin(angles[i]),1])
    p21 = np.array([positions[i] + 5, -14])
    p22 = p21 + 28 * np.array([math.sin(angles[i]), 1])
    pygame.draw.line(screen, white, cartesian_to_screen(p11),
                     cartesian_to_screen(p12), 3)
    pygame.draw.line(screen, white, cartesian_to_screen(p21),
                     cartesian_to_screen(p22), 3)
    # pygame.display.flip()

    display.blit(screen, (0, 0))
    pygame.display.update()

    # Convert the window in black color(2D) into a matrix
    screen_px = pygame.surfarray.array2d(display)
    screen_px = screen_px/ np.max(screen_px)

    screen_px = np.flip(np.rot90(np.rot90(np.rot90(screen_px))),axis=1)
    # time.sleep(0.1)
    # draw(screen_px)
    data[i] = screen_px
    # time.sleep(0.1)

x_train = data[:90000,:,:].astype('float32')
x_test = data[90000:,:,:].astype('float32')
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test),verbose=1)

#plot latent/hidden space

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
# plt.figure(figsize=(6, 6))
# plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
# plt.colorbar()
# plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

sliders =  np.zeros(latent_dim)
k=0
state = 'still'
while True:
    if state == 'up':
        sliders[k] +=0.04
    if state == 'down':
        sliders[k] -=0.04

    for event in pygame.event.get():
        # When click event
        if event.type == pygame.KEYDOWN:
            print(event.key)
            if event.key == pygame.K_LEFT:
                k-=1
            if event.key == pygame.K_RIGHT:
                k+=1
            if event.key == pygame.K_UP:
                state = 'up'
            if event.key == pygame.K_DOWN:
                state = 'down'
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                state = 'still'
            if event.key == pygame.K_DOWN:
                state = 'still'
    print(k, sliders[k])
    # draw(x_test[0].reshape(28,28))

    # time.sleep(1)
    z_sample = np.array([sliders])
    # z_sample = encoder.predict(x_test[0].reshape(1,784))
    x_decoded = generator.predict(z_sample)
    digit = x_decoded[0].reshape(digit_size, digit_size)
    draw(digit)
    # time.sleep(1)
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):

        print(digit)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()