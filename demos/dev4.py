import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split,show_images
from ezmodel.ezblocks4 import *
from ezmodel.eznetwork import reparam_trick_vae
from keras.models import Model
from ezmodel import ezlosses

import keras
import numpy as np

# [EZSET]  -------------------------------------------------------------------
parameters={
    "path"      : "C:\\Users\\daian\\Desktop\\DATA\\Mito\\images\\",
    "path_mask" : "C:\\Users\\daian\\Desktop\\DATA\\Mito\\masks\\",
    "resize"    : (128,128)
}
data = ezset(parameters)
data.autoencoder()

#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Transform
transformers = train.transform(X="standard",y="minmax")

# [EZNETWORK with custome EZBLOCKS]

deconv = Block().define(
"""
    UpSampling2D(size=(2,2))
    Conv2D(filters=?, kernel_size=(3,3), padding="same")
    BatchNormalization(momentum=0.8)
    Activation("relu")

"""
)

# GAN
import numpy as np
latent_dim = 100
img_shape  = train.X.shape[1:]
channels   = train.X.shape[3]
n_critic = 5
clip_value = 0.01

batch_size = 32
epochs = 50

#Generator
noise = Input(shape=(latent_dim,))
x = Dense(256 * 8 * 8, activation="relu") (noise)
x = Reshape((8, 8, 256)) (x)
x = deconv(filters=128) (x)
x = deconv(filters=64) (x)
x = deconv(filters=32) (x)
x = deconv(filters=16) (x)
x = Conv2D(channels, kernel_size=(4,4), padding="same") (x)
img = Activation("sigmoid") (x)
generator = Model(noise, img)
generator.summary()


#Discriminator


img = Input(shape=img_shape)
x = Conv2D(16, kernel_size=3, strides=2,padding="same") (img)
x = LeakyReLU(alpha=0.2) (x)
x = Dropout(0.25) (x)

x = Conv2D(32, kernel_size=3, strides=2, padding="same") (x)
x = ZeroPadding2D(padding=((0,1),(0,1))) (x)
x = BatchNormalization(momentum=0.8) (x)
x = LeakyReLU(alpha=0.2) (x)
x = Dropout(0.25) (x)

x = Conv2D(64, kernel_size=3, strides=2, padding="same") (x)
x = BatchNormalization(momentum=0.8) (x)
x = LeakyReLU(alpha=0.2) (x)
x = Dropout(0.25) (x)

x = Conv2D(128, kernel_size=3, strides=1, padding="same") (x)
x = BatchNormalization(momentum=0.8) (x)
x = LeakyReLU(alpha=0.2) (x)
x = Dropout(0.25) (x)

x = Flatten() (x)
validity = Dense(1) (x)
discriminator = Model(img, validity)

discriminator.summary()

# Build and compile the critic
discriminator.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
                      loss=ezlosses.wasserstein_loss,
                      metrics=['accuracy']
                      )
# For the combined model we will only train the generator
discriminator.trainable = False

#Gan : combined (gen + disc)
z = Input(shape=(latent_dim,))
img = generator(z)
valid = discriminator(img)
# The combined model  (stacked generator and critic)
gan = Model(z, valid)
gan.compile(optimizer=keras.optimizers.Adam(lr=1e-4),loss=ezlosses.wasserstein_loss,metrics=['accuracy'])

#Training
#Adversarial GT
valid = -np.ones((batch_size, 1))
fake = np.ones((batch_size, 1))

input0 = copy.deepcopy(train)
input0.preprocess(X=transformers[0],y=transformers[1])


for epoch in range(epochs):
    for _ in range(n_critic):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, train.X.shape[0], batch_size)
        # imgs = train.X[idx]
        imgs = input0.X[idx]

        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Generate a batch of new images
        gen_imgs = generator.predict(noise)

        # Train the critic
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

        # Clip critic weights
        for l in discriminator.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -clip_value, clip_value) for w in weights]
            l.set_weights(weights)

    # ---------------------
    #  Train Generator
    # ---------------------

    g_loss = gan.train_on_batch(noise, valid)

    # Plot the progress
    print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))


import matplotlib.pyplot as plt
r, c = 5, 5
noise = np.random.normal(0, 1, (r * c, latent_dim))
gen_imgs = generator.predict(noise)

# Rescale images 0 - 1
gen_imgs = 0.5 * gen_imgs + 0.5

fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
        axs[i,j].axis('off')
        cnt += 1
plt.show()

#
