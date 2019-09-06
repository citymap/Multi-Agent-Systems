from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import pygame
from keras import objectives


input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded_conv = MaxPooling2D((2, 2), padding='same')(x)
print(encoded_conv)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
print(decoded)
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder_conv = Model(input_img, encoded_conv)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(4,4,8))
first = autoencoder.layers[-7](encoded_input)
second = autoencoder.layers[-6](first)
third = autoencoder.layers[-5](second)
fourth = autoencoder.layers[-4](third)
fifth = autoencoder.layers[-3](fourth)
sixth = autoencoder.layers[-2](fifth)
seventh = autoencoder.layers[-1](sixth)

decoder = Model(encoded_input, seventh)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

image_size = 28
original_dim = image_size * image_size

# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 3
epochs = 5

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# end-to-end autoencoder
vae = Model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder = Model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

vae.compile(optimizer='rmsprop', loss=vae_loss)


from keras.datasets import mnist
import numpy as np
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
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format


# (x_train, _), (x_test, _) = mnist.load_data()
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
# x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

encoded_imgs = encoder.predict(x_test)
print(encoded_imgs)
decoded_imgs = decoder.predict(encoded_imgs)
# decoded_imgs = autoencoder.predict(x_test)
import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()