import pickle
from zipfile import ZipFile
import os
import time
from keras.models import load_model
import sys
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,roc_auc_score,precision_score,recall_score,precision_recall_curve,f1_score,average_precision_score

import copy
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from decimal import Decimal

from ezmodel.ezblocks4 import *
from keras.models import Model


class ezgan:

    def __init__(self,data=None,transformers=None,parameters=None):
        self.data = data
        self.transformers = transformers
        self.parameters = parameters
        self.generator = None
        self.discriminator = None
        self.gan = None
        keys = ["disc_loss","disc_acc","disc_acc_real","disc_acc_fake","gen_loss"]
        self.history = {key: [] for key in keys}

    def basic_generator(self,data,parameters=None):

        if parameters is None:
            #Default parameters
            parameters = {"noise_dim" : 100,"n" : 256,"depth"     : 4}
            print("[ ] Basic Generator:")
            print("--- Default parameters has been choosen : ", parameters)

        #Check wether depth and image size are consistent
        initial_size = int(data.X.shape[1]/(2**parameters["depth"]))
        if initial_size<1:
            raise Exception('ezgan.basic_generator(): Depth too large for your Image size !')
        #Generator
        #Ezblock
        deconv = Block().define(
        """
            UpSampling2D(size=(2,2))
            Conv2D(filters=?, kernel_size=(3,3), padding="same")
            BatchNormalization(momentum=0.8)
            Activation("relu")

        """
        )
        noise = Input(shape=(parameters["noise_dim"],))
        x = Dense(parameters["n"] * initial_size * initial_size, activation="relu") (noise)
        x = Reshape((initial_size, initial_size, parameters["n"])) (x)
        for i in range(parameters["depth"]):
            x = deconv(filters=int(parameters["n"]/(2**i))) (x)
        x = Conv2D(data.X.shape[3], kernel_size=(4,4), padding="same") (x)
        img = Activation("sigmoid") (x)
        generator = Model(noise, img)
        self.generator = generator
        self.generator.summary()
        print("[X] Basic generator created ! ")

    def basic_discriminator(self,data,parameters=None):
        if parameters is None:
            #Default parameters
            parameters = {"depth" : 4, "n": 128}
            print("[ ] Basic Discriminator:")
            print("--- Default parameters has been choosen : ", parameters)

            #Discriminator

        conv = Block().define(
        """
            Conv2D(filters=?,kernel_size=(3,3),strides=2,padding="same")
            BatchNormalization(momentum=0.8)
            LeakyReLU(alpha=0.2)
            Dropout(0.25)
        """
        )

        img_shape = self.generator.output_shape[1:]
        img = Input(shape=img_shape)
        x = img
        # for i in reversed(range(parameters["depth"])):
            # x = conv(filters=int(img_shape[0]*(2**(parameters["depth"]-i)))) (x)
        # step = int(parameters["n"] / (2**(parameters["depth"]-1)))
        # print(step)
        for i in reversed(range(parameters["depth"])):
            step = int(parameters["n"]/(2**(i+1)))
            x = conv(filters=step) (x)


        x = Flatten() (x)
        validity = Dense(1) (x)
        self.discriminator = Model(img, validity)
        self.discriminator.summary()

        print("[X] Basic Discriminator created ! ")

    def generate(self,optimizer=None):

        if optimizer is None:
            raise Exception('[Fail] ezgan.generate : Please provide one optimizer !')

        latent_dim = self.generator.input_shape[1]
        z = Input(shape=(latent_dim,))
        img = self.generator(z)
        valid = self.discriminator(img)
        # The combined model  (stacked generator and critic)
        gan = Model(z, valid)

        #Models Compilations
        # Discriminator + Freeze
        self.discriminator.compile(**optimizer)
        self.discriminator.trainable = False

        # Gan
        gan.compile(**optimizer)
        self.gan = gan
        self.gan.summary()

        print("[X] GAN created ! ")


    def train(self,parameters=None):
        if parameters is None:
            parameters = {
                "epochs": 10,
                "batch_size": 32,
                "n_discriminator": 7,
                "discriminator_clip_value" : 0.01,
                "logdir": None
            }
            print("[ ] GAN Training:")
            print("--- Default parameters has been choosen : ", parameters)

        batch_size = parameters["batch_size"]
        epochs = parameters["epochs"]
        n_critic = parameters["n_discriminator"]
        clip_value = parameters["discriminator_clip_value"]

        valid = -1*np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        input0 = copy.deepcopy(self.data)
        input0.preprocess(X=self.transformers[0],y=None)


        for epoch in range(epochs):
            for _ in range(n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, self.data.X.shape[0], batch_size)
                # imgs = train.X[idx]
                imgs = input0.X[idx]

                # Sample noise as generator input
                latent_dim = self.generator.input_shape[1]
                noise = np.random.normal(0, 1, (batch_size, latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.gan.train_on_batch(noise, valid)

            # Plot the progress
            print ("Epoch %d : [Discriminator loss: %f, acc: %f, acc on Real: %f, acc on Fake: %f] [Generator loss: %f] " % (epoch, 1 - d_loss[0],d_loss[1],d_loss_real[1],d_loss_fake[1], 1 - g_loss[0]))

            #History
            self.history["disc_loss"].append(1-d_loss[0])
            self.history["disc_acc"].append(d_loss[1])
            self.history["disc_acc_real"].append(1-d_loss_real[0])
            self.history["disc_acc_fake"].append(1-d_loss_fake[0])
            self.history["gen_loss"].append(1-g_loss[0])

            if parameters["logdir"] is not None:
                self.plotGeneratedImages(epoch,logdir=parameters["logdir"])



    def plotGeneratedImages(self,epoch, logdir=None, examples=25, dim=(5, 5), figsize=(20, 20)):
        noise = np.random.normal(0, 1, size=[examples, self.generator.input_shape[1]])
        generatedImages = self.generator.predict(noise)

        plt.figure(figsize=figsize)
        for i in range(generatedImages.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)

            if generatedImages[i].shape[2]==1:
                plt.imshow(np.squeeze(generatedImages[i]),cmap="gray")
                cmap="gray"
            else:
                plt.imshow(generatedImages[i])
            plt.axis('off')

        plt.tight_layout()

        if logdir is not None:
            plt.savefig(os.path.join(logdir,'gan_epoch_'+str(epoch)+'.png'))

        plt.close()
