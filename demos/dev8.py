import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split,show_images
from ezmodel.ezblocks4 import *
from ezmodel.eznetwork import reparam_trick_vae
from keras.models import Model, Sequential
from ezmodel import ezlosses
from ezmodel.ezgan import ezgan
from ezmodel.ezblocks4 import *

import keras
import numpy as np

# [EZSET]  -------------------------------------------------------------------
parameters={
    "path"      : "C:\\Users\\daian\\Desktop\\DATA\\Mito\\images\\",
    "path_mask" : "C:\\Users\\daian\\Desktop\\DATA\\Mito\\masks\\",
    "resize"    : (128,128)
}
data = ezset(parameters)
transformers = data.transform(X="minmax",y=None)


# [EZGAN] ---------------------------------------------------------------------
# gan = ezgan(
#         data=data,
#         transformers=transformers,
#         parameters=parameters
# )

# GAN
latent_dim = 100
img_shape  = data.X.shape[1:]
channels   = data.X.shape[3]
n_critic = 5
clip_value = 0.01

#Generator
deconv = Block().define(
"""
    UpSampling2D(size=(2,2))
    Conv2D(filters=?, kernel_size=(3,3), padding="same")
    BatchNormalization(momentum=0.8)
    Activation("relu")

"""
)


noise = Input(shape=(latent_dim,))
x = Dense(1024 * 8 * 8, activation="relu") (noise)
x = Reshape((8, 8, 1024)) (x)
x = deconv(filters=512) (x)
x = deconv(filters=256) (x)
x = deconv(filters=128) (x)
x = deconv(filters=64) (x)
x = Conv2D(channels, kernel_size=(4,4), padding="same") (x)
img = Activation("sigmoid") (x)
generator = Model(noise, img)
generator.summary()
# gan.generator = generator

#Discriminator


img = Input(shape=img_shape)
x = Conv2D(32, kernel_size=3, strides=2,padding="same") (img)
x = LeakyReLU(alpha=0.2) (x)
x = Dropout(0.25) (x)

x = Conv2D(64, kernel_size=3, strides=2, padding="same") (x)
x = ZeroPadding2D(padding=((0,1),(0,1))) (x)
x = BatchNormalization(momentum=0.8) (x)
x = LeakyReLU(alpha=0.2) (x)
x = Dropout(0.25) (x)

x = Conv2D(128, kernel_size=3, strides=2, padding="same") (x)
x = BatchNormalization(momentum=0.8) (x)
x = LeakyReLU(alpha=0.2) (x)
x = Dropout(0.25) (x)

x = Conv2D(256, kernel_size=3, strides=2, padding="same") (x)
x = BatchNormalization(momentum=0.8) (x)
x = LeakyReLU(alpha=0.2) (x)
x = Dropout(0.25) (x)

x = Conv2D(512, kernel_size=3, strides=1, padding="same") (x)
x = BatchNormalization(momentum=0.8) (x)
x = LeakyReLU(alpha=0.2) (x)
x = Dropout(0.25) (x)


x = Flatten() (x)
validity = Dense(1) (x)
discriminator = Model(img, validity)

discriminator.summary()

#
#
#
# img = Input(shape=img_shape)
# x = Conv2D(64, kernel_size=3, strides=2,padding="same") (img)
# x = LeakyReLU(alpha=0.2) (x)
# x = Dropout(0.25) (x)
#
# x = Conv2D(128, kernel_size=3, strides=2, padding="same") (x)
# x = ZeroPadding2D(padding=((0,1),(0,1))) (x)
# x = BatchNormalization(momentum=0.8) (x)
# x = LeakyReLU(alpha=0.2) (x)
# x = Dropout(0.25) (x)
#
# x = Conv2D(256, kernel_size=3, strides=2, padding="same") (x)
# x = BatchNormalization(momentum=0.8) (x)
# x = LeakyReLU(alpha=0.2) (x)
# x = Dropout(0.25) (x)
#
# x = Conv2D(512, kernel_size=3, strides=1, padding="same") (x)
# x = BatchNormalization(momentum=0.8) (x)
# x = LeakyReLU(alpha=0.2) (x)
# x = Dropout(0.25) (x)
#
# x = Flatten() (x)
# validity = Dense(1) (x)
# discriminator = Model(img, validity)
#
# discriminator.summary()

# Build and compile the critic
discriminator.compile(
                      # optimizer=keras.optimizers.Adam(lr=1e-4),
                      keras.optimizers.RMSprop(lr=0.00005),
                      loss=ezlosses.wasserstein_loss,
                      # loss = keras.losses.binary_crossentropy,
#                       loss = keras.losses.binary_crossentropy,
                      metrics=['accuracy']
                      )
# For the combined model we will only train the generator
discriminator.trainable = False
# gan.discriminator = discriminator


#Gan : combined (gen + disc)
z = Input(shape=(latent_dim,))
img = generator(z)
valid = discriminator(img)
# The combined model  (stacked generator and critic)
GAN = Model(z, valid)
GAN.compile(
          # optimizer=keras.optimizers.Adam(lr=1e-4),
          optimizer = keras.optimizers.RMSprop(lr=0.00005),
          loss=ezlosses.wasserstein_loss,
          # loss = keras.losses.binary_crossentropy,
#           loss = keras.losses.binary_crossentropy,
          metrics=['accuracy'])

# gan.gan = GAN



batch_size = 32
epochs = 1000
logdir = "C:\\Users\\daian\\Desktop\\LOGDIR\\GAN-MITO\\"

import matplotlib.pyplot as plt


#Training
#Adversarial GT
valid = -np.ones((batch_size, 1))
fake = np.ones((batch_size, 1))

input0 = copy.deepcopy(data)
input0.preprocess(X=transformers[0],y=transformers[1])



for epoch in range(epochs):
    for _ in range(n_critic):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, data.X.shape[0], batch_size)
        # imgs = train.X[idx]
        imgs = input0.X[idx]

        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Generate a batch of new images
        # gen_imgs = gan.generator.predict(noise)
        gen_imgs = generator.predict(noise)

        # Train the critic
        # d_loss_real = gan.discriminator.train_on_batch(imgs, valid)
        # d_loss_fake = gan.discriminator.train_on_batch(gen_imgs, fake)
        # d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

        # Clip critic weights
        # for l in gan.discriminator.layers:
        for l in discriminator.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -clip_value, clip_value) for w in weights]
            l.set_weights(weights)

    # ---------------------
    #  Train Generator
    # ---------------------

    # g_loss = gan.gan.train_on_batch(noise, valid)
    g_loss = GAN.train_on_batch(noise, valid)

    # Plot the progress
#     print ("%d [D loss: %f] [D acc: %f] [G loss: %f] " % (epoch, 1 - d_loss[0],d_loss[1], 1 - g_loss[0]))
    print ("%d [D loss: %f] [D acc: %f] [D acc on Real: %f] [D acc on Fake: %f] [G loss: %f] " % (epoch, 1 - d_loss[0],d_loss[1],d_loss_real[1],d_loss_fake[1], 1 - g_loss[0]))

    if epoch % 10 == 0:
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        # gen_imgs = gan.generator.predict(noise)
        gen_imgs = generator.predict(noise)

        # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
#                 axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
              if gen_imgs[cnt].shape[2]==1:
                axs[i,j].imshow(np.squeeze(gen_imgs[cnt]),cmap="gray")
              else:
                axs[i,j].imshow(gen_imgs[cnt])

              axs[i,j].axis('off')
              cnt += 1
        # plt.show()
        plt.savefig(os.path.join(logdir,'gan_epoch'+str(epoch)+'.png'))
        plt.close()
#FIRST

# # Generator -------------------------------------------------------------------
# parameters = {
#     "noise_dim" : 100,
#     "n" : 256,
#     "depth"     : 4
# }
# gan.basic_generator(data=data,parameters=parameters)
#
# # Discriminator ---------------------------------------------------------------
# parameters = {
#     "n": 128,
#     "depth": 4,
# }
# gan.basic_discriminator(data=data,parameters=parameters)
#
# # Gan -------------------------------------------------------------------------
# optimizer_gan = {
#     "optimizer" : keras.optimizers.RMSprop(lr=0.0005),
#     "loss"      : ezlosses.wasserstein_loss,
#     "metrics"   : ["accuracy"]
# }
# gan.generate(optimizer=optimizer_gan)
#
# # Training
# parameters = {
#     "epochs": 1000,
#     "batch_size": 32,
#     "n_discriminator": 7,
#     "discriminator_clip_value" : 0.01,
#     "logdir" : "C:\\Users\\daian\\Desktop\\LOGDIR\\GAN-POKE\\"
# }
# gan.train(parameters)
