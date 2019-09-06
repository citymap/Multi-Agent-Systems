# Author:   Max Martinez Ruts
# Creation: 2019

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
import pygame
import time

def wt(img):
    wait = True
    drawing = False
    while wait:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos

                # Record mouse position
                if drawing:
                    for i in range(0,28):
                        for j in range(0,28):
                            pixel = pygame.Rect(i * width / 28, j * width / 28, width / 28, width / 28)
                            if pixel.collidepoint(mouse_pos):
                                img[j,i] = 0.

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                # for i in range(0, 28):
                #     for j in range(0, 28):
                #         pixel = pygame.Rect(i * width / 28, j * width / 28, width / 28, width / 28)
                #         if pixel.collidepoint(mouse_pos):
                #             for k in range(28):
                #                 for l in range(28):
                #                     if i - 3 <= k <= i + 3 and j - 3 <= l <= j + 3:
                #                         img[k,l] = 2.0

                # Record mouse position
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                # Record mouse position
                drawwing = False
                wait = False
        draw(img)
    return img

# Convert coordinates form cartesian to screen coordinates (used to draw in pygame screen)
def cartesian_to_screen(car_pos):
    factor = 1
    screen_pos = np.array([center[0]*factor+car_pos[0],center[1]*factor+car_pos[1]])/factor
    screen_pos = screen_pos.astype(int)
    return screen_pos



def draw(img):
    pygame.event.get()
    screen.fill((0, 0, 0))
    for i in range(0, 28):
        for j in range(0, 28):
            color = min(int(img[j,i]*255),255)
            cl = (color,color,color)

            if int(img[j,i]*255) > 255:
                cl = (0,255,0)
            pixel = pygame.Rect(i*width/28,j*width/28, width/28, width/28)
            pygame.draw.rect(screen, cl, pixel)

    pygame.display.flip()

# Screen parameters
width = 800
height = 1000
center = np.array([width/2, height/2])
screen = pygame.display.set_mode((width, height))
pixel_pos = np.arange(0, width, width/28)

sliders = np.zeros(32)

# this is the size of our encoded representations
encoding_dim = 100  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
print(x_train.shape)

x_train_partial = np.array(list(x_train)).reshape(60000,28,28)
x_test_partial = np.array(list(x_test)).reshape(10000,28,28)

for i in range(x_train_partial.shape[0]):
    r1 = np.random.uniform(0, 28)
    r2 = np.random.uniform(0, 28)
    for j in range(28):
        for k in range(28):
            if r1-3<=j<=r1+3 and r2-3<=k<=r2+3:
                x_train_partial[i,j,k] = np.random.uniform(0,1)

for i in range(x_test_partial.shape[0]):
    r1 = np.random.uniform(0, 28)
    r2 = np.random.uniform(0, 28)
    for j in range(28):
        for k in range(28):
            if r1-3<=j<=r1+3 and r2-3<=k<=r2+3:
                x_test_partial[i,j,k] = np.random.uniform(0,1)

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
# fig = plt.figure()
#
plt.imshow(x_train[0].reshape(28, 28))
plt.gray()
plt.show()

plt.imshow(x_train_partial[0])
plt.gray()
plt.show()
print(x_train_partial.flatten().shape, x_train_noisy.shape)

autoencoder.fit(x_train_partial.reshape(60000,784), x_train,
                epochs=30,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_partial.reshape(10000,784), x_test))

# x_test[0] =np.zeros((28,28), dtype=float)

while True:
    real = wt(x_test[0].reshape(28,28))
    print('done')
    encoded_imgs = encoder.predict(real.reshape(1,784))
    decoded_imgs = decoder.predict(encoded_imgs)
    draw(decoded_imgs[0].reshape(28, 28))
    time.sleep(3)


# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test_partial.reshape(10000,784))
decoded_imgs = decoder.predict(encoded_imgs)

n = 30  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_partial.reshape(10000,784)[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display original
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test[i].reshape(28, 28))
    draw(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    draw(decoded_imgs[i].reshape(28, 28))

    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()

