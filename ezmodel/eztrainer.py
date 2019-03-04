from sklearn.model_selection import train_test_split
from keras.models import Input,Model, Sequential
from keras.layers import *
from keras import layers
import keras
import keras.backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121,DenseNet169,DenseNet201
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import NASNetLarge,NASNetMobile
from keras.applications.resnet50 import ResNet50

import matplotlib.pyplot as plt
import numpy as np
import math


class eztrainer:

    def __init__(self):

        self.network = None
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.history = None
        self.image_aug = None
        self.transfer = None

    def gen_trainval(self,ezdata,size=0.2,random_state=42):

        self.X_train,self.X_valid,self.y_train,self.y_valid = train_test_split(ezdata.X,ezdata.y,test_size=size,random_state=42)

        print ("[X] Train/Validation set generation (size = ",str(size),"): Done")
        print ("--- Training set : ", self.X_train.shape[0], "images")
        print ("--- Validation set     : ", self.X_valid.shape[0], "images")
        print("\n")

    def keras_augmentation(self,parameters):
        image_gen = ImageDataGenerator(**parameters)
        if self.X_train is None:
            raise Exception("[Fail] eztrainer.image_augmentation(): Training set has not been generated yet ! Please use 'gen_trainval() on eztrainer object before'")
        image_gen.fit(self.X_train, augment=True)
        self.image_aug = image_gen
        print("[X] Keras ImageDataGenerator has been added to eztrainer object")
        print("\n")



    # def show_images(self,n=16):
    #
    #     if not math.sqrt(n).is_integer():
    #         raise Exception("[Fail] ezdata.show_images(): Please provide n as a perfect quare ! (2, 4, 9, 16, 25, 36, 49, 64 ...)")
    #
    #     augmented_images, _ = next( self.image_aug.flow( self.X_train, self.y_train, batch_size=n))
    #
    #     fig,axes = plt.subplots(nrows = int(math.sqrt(n)),ncols = int(math.sqrt(n)))
    #     fig.tight_layout()
    #     for i in range(n):
    #         plt.subplot(math.sqrt(n),math.sqrt(n),i+1)
    #
    #         if (augmented_images[i].shape[2])==1:
    #             plt.imshow(np.squeeze(augmented_images[i]),cmap="gray")
    #         else:
    #             plt.imshow(augmented_images[i])
    #
    #         plt.axis("off")
    #
    #     plt.show()




    def gen_network(self,inputs,outputs,transfer_model=None):
        if transfer_model is None:
            m = Model(inputs=inputs,outputs=outputs)
            self.network = m
        else:
            bottom = Model(inputs=inputs,outputs=outputs)
            global_model = Sequential()
            global_model.add(transfer_model)
            global_model.add(bottom)
            self.network = global_model

    def Transfer(self,name,frozen=True,include_top=False):

        if name.lower()=="mobilenet":
            transfer_network = MobileNet(include_top=include_top, weights='imagenet', input_shape=self.X_train.shape[1:])
            self.transfer = "mobilenet"

        if name.lower()=="vgg16":
            transfer_network = VGG16(include_top=include_top, weights='imagenet', input_shape=self.X_train.shape[1:])
            self.transfer = "vgg16"

        if name.lower()=="vgg19":
            transfer_network = VGG19(include_top=include_top, weights='imagenet', input_shape=self.X_train.shape[1:])
            self.transfer = "vgg19"


        if frozen:
            for layer in transfer_network.layers:
                    layer.trainable = False

        model = Sequential()
        model.add(transfer_network)
        model.add(GlobalAveragePooling2D())

        return model


    def Input(self,transfer_model=None):
        if transfer_model is None:
            return Input(shape=self.X_train.shape[1:])
        else:
            inputs = Input(shape = transfer_model.output_shape[1:])
            return inputs


    def ClassificationOutput(self,x0):

        if len(self.y_train.shape)==1:
            if len(np.unique(self.y_train))==2: #probably classification with binary class
                x = layers.Dense(1) (x0)
                x = layers.Activation("sigmoid") (x)
                return x
            else: #probably classification multiclass  or regression
                x = layers.Dense(1) (x0)
                x = layers.Activation("linear") (x)
                return x

        if len(self.y_train.shape)==2: #probably classification with multiclass
            x = layers.Dense(self.y_train.shape[1])(x0)
            x = layers.Activation("softmax") (x)
            return x

    def compile(self,optimizer):
            if hasattr(self,"network"):
                self.network.compile(**optimizer)
            else:
                raise Exception("[Fail] compile() : No network to compile the optimizer with. Please use gen_network() on your Keras network before.")

    def Network(self,name=None,parameters=None,transfer=False):
        if name is None:
            raise Exception("[Fail] eztrainer.Network(): Please enter a name for a prebuilt neural network")

        if name.lower() == "unet":
            self.network = self.Network_UNET(parameters)

        if name.lower() == "lenet5":
            self.network = self.Network_LENET5(parameters)

        #if name.lower() == "mobilenet":


    def Network_LENET5(self,parameters):

        #  -- Keras network --
        inputs = self.Input()
        x = Conv2D(6, kernel_size = (5, 5), strides=(1,1), padding="valid",input_shape=(32, 32, 1)) (inputs)
        x = Activation("relu") (x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (x)
        x = Conv2D(16, kernel_size = (5, 5), strides=(1,1), padding="valid") (x)
        x = Activation("relu") (x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (x)
        x = Flatten() (x)
        x = Dense(120) (x)
        x = Dense(84) (x)
        outputs = self.ClassificationOutput(x)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model



    # UNET network
    def Network_UNET(self,parameters):

        #Default start number of filters
        n = 64
        if parameters is not None:
            if "n" in parameters:
                n = parameters[n]


        inputs = self.Input()
        conv1 = Conv2D(n, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(n, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(n*2, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(n*2, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(n*4, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(n*4, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(n*8, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(n*8, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(n*16, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(n*16, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(n*8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(n*8, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(n*8, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(n*4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(n*4, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(n*4, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(n*2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(n*2, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(n*2, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(n, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(n, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(n, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])
        return model



#EZOptimizer ...
class ezoptimizer:


    def _init__(self,name):
        pass

    def loss(self,name=None):
        if name is None:
            raise Exception("[Fail] eztrainer.ezoptimizer.loss() : Please provide a loss name for the optimizer")

        if name == "dice_coefficient":
            return self.dice_coef_loss

    def metrics(self,name=None):
        if name is None:
            raise Exception("[Fail] eztrainer.ezoptimizer.metrics() : Please provide a metrics name for the optimizer")

        if name == "dice_coefficient":
            return [self.dice_coef]


    def dice_coef(self,y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


    def dice_coef_loss(self,y_true, y_pred):
      return -self.dice_coef(y_true, y_pred)
