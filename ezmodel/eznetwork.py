from keras.models import *
from keras.layers import *
from keras.applications import *

from ezmodel.ezblocks4 import *

import keras


import os
import copy



def SmartInput(input):
    if len(input.X.shape)==4:
        return Input(shape=input.X.shape[1:])
    if len(input.X.shape)==2:
        return Input(shape=(input.X.shape[1],))

def SmartClassificationRegressionOutput(input,x0):
    if len(input.y.shape)==1:
        if len(np.unique(input.y))==2: #probably classification with binary class
            x = Dense(1) (x0)
            x = Activation("sigmoid") (x)
            return x
        else: #probably classification multiclass  or regression
            x = Dense(1) (x0)
            x = Activation("linear") (x)
            return x

    if len(input.y.shape)==2: #probably classification with multiclass
        x = Dense(input.y.shape[1])(x0)
        x = Activation("softmax") (x)
        return x

def SmartClassificationRegressionOutputSequential(input):
    if len(input.y.shape)==1:
        if len(np.unique(input.y))==2: #probably classification with binary class
            x = Dense(1,activation="sigmoid")
            return x
        else: #probably classification multiclass  or regression
            x = Dense(1,activation="linear")
            return x

    if len(input.y.shape)==2: #probably classification with multiclass
        x = Dense(input.y.shape[1],activation="softmax")
        return x

#Load a pretrained Network
def Pretrained(input=None,path=None,include_top=False,transfer=False,frozen=False):
    if path.lower()=="mobilenet":
        pretrained = mobilenet.MobileNet(include_top=include_top, weights='imagenet', input_shape=input.X.shape[1:])
    elif path.lower()=="mobilenetv2":
        pretrained = mobilenet_v2.MobileNetV2(include_top=include_top, weights='imagenet', input_shape=input.X.shape[1:])
    elif path.lower()=="vgg16":
        pretrained = vgg16.VGG16(include_top=include_top, weights='imagenet', input_shape=input.X.shape[1:])
    elif path.lower()=="vgg19":
        pretrained = vgg19.VGG19(include_top=include_top, weights='imagenet', input_shape=input.X.shape[1:])
    elif path.lower()=="xception":
        pretrained = xception.Xception(include_top=include_top, weights='imagenet', input_shape=input.X.shape[1:])
    elif path.lower()=="inceptionv3":
        pretrained = inception_v3.InceptionV3(include_top=include_top, weights='imagenet', input_shape=input.X.shape[1:])
    else:
        #check local model
        if os.path.isfile(path+".h5"):
            pretrained = load_model(path+".h5")
        else:
            raise Exception("eznetwork.Pretrained(): No model path: ",path," has been found !")

    if frozen:
        for layer in pretrained.layers:
                layer.trainable = False

    return pretrained

def Connect(bottom,top):
    model = Sequential()
    model.add(bottom)
    model.add(GlobalAveragePooling2D())
    model.add(top)
    return model


#LeNet5
def LeNet5(input=None, transformers=None, parameters=None):

    #Checkers:
    if len(input.X.shape) !=4:
        raise Exception("\n\n \t [Fail] eznetwork.MLP(): LeNet5 convnet is not designed to work with this kind of inputs ! Please use another Network architecture ! ")

    #Temporary transform data
    if transformers is not None:
        input0 = copy.deepcopy(input)
        input0.preprocess(X=transformers[0],y=transformers[1])
    else:
        input0 = input

    inputs = SmartInput(input0)
    x = Conv2D(6, kernel_size = (5, 5), strides=(1,1), padding="valid") (inputs)
    x = Activation("relu") (x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (x)
    x = Conv2D(16, kernel_size = (5, 5), strides=(1,1), padding="valid") (x)
    x = Activation("relu") (x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (x)
    x = Flatten() (x)
    x = Dense(120) (x)
    x = Dense(84) (x)
    outputs = SmartClassificationRegressionOutput(input0,x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# [EZNETWORK]  ----------------------------------------------------------------
def reparam_trick_vae(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))

    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#Basic Variational Autoencoder
def VAE(input=None,transformers=None,parameters=None,optimizer = None):

    #Temporary transform data
    if transformers is not None:
        input0 = copy.deepcopy(input)
        input0.preprocess(X=transformers[0],y=transformers[1])
    else:
        input0 = input

    intermediate_dim = parameters["intermediate_dim"]
    activation       = parameters["activation"]
    latent_dim       = parameters["latent_dim"]

    inputs = SmartInput(input0)

    #Encoder
    x = Dense(intermediate_dim, activation=activation)(inputs)
    if "bn" in parameters:
        x = BatchNormalization(momentum=parameters["bn"])(x)
    if "dropout" in parameters:
        x = Dropout(parameters["dropout"]) (x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(reparam_trick_vae, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    #Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    if "bn" in parameters:
        x = BatchNormalization(momentum=parameters["bn"])(x)
    if "dropout" in parameters:
        x = Dropout(parameters["dropout"])(x)
    outputs = Dense(encoder.input_shape[1], activation='linear')(x) #Test du activation output = linear
    decoder = Model(latent_inputs, outputs, name='decoder')

    outputs = decoder(encoder(inputs)[2]) #We only take the latent z as input of the decoder that's why we use [2]
    vae = Model(inputs, outputs, name='vae_mlp')
    #
    # #Embed compilation
    # if optimizer is not None:
    #     #Recontruction_loss
    #     reconstruction_loss = keras.losses.mse(inputs,outputs)
    #     reconstruction_loss *= encoder.input_shape[1]
    #     #KL-divergeance
    #     kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    #     kl_loss = K.sum(kl_loss, axis=-1)
    #     kl_loss *= -0.5
    #     #VAE loss as combination of reconstruction_loss and KL-divergence
    #     vae_loss = K.mean(reconstruction_loss + kl_loss)
    #     vae.add_loss(vae_loss)
    #     vae.compile(optimizer=optimizer["optimizer"])
    #     print("[X] VAE has been compiled with Reconstruction Loss and KL-Divergence")


    return (vae,z_mean,z_log_var)

#Basic Multilayer Perceptron
def MLP(input=None, transformers=None, parameters=None,pretrained=None):

    #Checkers:
    if (input is not None) and (pretrained is None):
        if len(input.X.shape) !=2:
            raise Exception("\n\n \t [Fail] eznetwork.MLP(): Multi Layer Perceptron are not designed to work with this kind of inputs ! Please use another Network architecture ! ")
    if parameters is None:
        raise Exception("\n\n \t [Fail] eznetwork.MLP(): Please provide a parameters list to MLP ! ")

    #Temporary transform data
    if transformers is not None:
        input0 = copy.deepcopy(input)
        input0.preprocess(X=transformers[0],y=transformers[1])
    else:
        input0 = input

    if "hidden" not in parameters:
        raise Exception("\n\n\t [Fail] eznetwork.MLP(): MLP with No Layer ! Please provide at leat one layer")
    else:
        if type(parameters["hidden"]) is int:
            hidden = [parameters["hidden"]]
        elif type(parameters["hidden"]) is list:
            hidden = parameters["hidden"]
        else:
            raise Exception("\n\n\t [Fail] eznetwork.MLP() : Please provide hidden layer parameters as a python list !")

    if pretrained is None:
        inputs = SmartInput(input0)
    else:
        #inputs = Input(shape=(pretrained.output_shape[3],))
        inputs = Input(shape=(pretrained.output_shape[3],))
        #inputs = GlobalAveragePooling2D()(inputs)

    x = Dense(hidden[0]) (inputs)
    if "activation" in parameters:
        x = Activation(parameters["activation"]) (x)
    if "dropout" in parameters:
        x = Dropout(parameters["dropout"]) (x)
    for layer in hidden[1:]:
        x = Dense(layer) (x)
        if "activation" in parameters:
            x = Activation(parameters["activation"]) (x)
        if "dropout" in parameters:
            x = Dropout(parameters["dropout"]) (x)
    outputs = SmartClassificationRegressionOutput(input0,x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# UNET network for segmentation
def UNET(input=None,transformers=None,parameters=None):
    #Default start number of filters
    n = 64
    if parameters is not None:
        if "n" in parameters:
            n = parameters["n"]

    inputs = SmartInput(input)
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

    model = Model(inputs=inputs, outputs=conv10)
    return model

def SmallAlexNet(input=None,transformers=None,parameters=None):

    #Temporary transform data
    if transformers is not None:
        input0 = copy.deepcopy(input)
        input0.preprocess(X=transformers[0],y=transformers[1])
    else:
        input0 = input

    inputs = SmartInput(input0)
    x = Conv2D(filters=96, kernel_size=(11,11),padding='valid',strides=(2,2)) (inputs)
    x = BatchNormalization() (x)
    x = Activation('relu') (x)
    x = MaxPooling2D(pool_size=(2,2)) (x)

    x = Conv2D(filters=256, kernel_size=(5,5), padding='valid',strides=(1,1)) (x)
    x = BatchNormalization() (x)
    x = Activation('relu') (x)
    x = MaxPooling2D(pool_size=(2,2)) (x)

    x = Conv2D(filters=384, kernel_size=(3,3), padding='valid') (x)
    x = BatchNormalization() (x)
    x = Activation('relu') (x)

    x = Conv2D(filters=384, kernel_size=(3,3), padding='valid') (x)
    x = BatchNormalization() (x)
    x = Activation('relu') (x)

    x = Conv2D(filters=256, kernel_size=(3,3), padding='valid') (x)
    x = BatchNormalization() (x)
    x = Activation('relu') (x)
    x = MaxPooling2D(pool_size=(2,2)) (x)

    x = Flatten() (x)

    x = Dense(1024) (x)
    x = Activation('relu') (x)
    x = Dropout(0.4) (x)
    x = BatchNormalization() (x)

    x = Dense(1024) (x)
    x = Activation('relu') (x)
    x = Dropout(0.4) (x)
    x = BatchNormalization() (x)

    outputs = SmartClassificationRegressionOutput(input0,x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def InceptionV3(input=None,transformers=None,parameters=None):

    #Temporary transform data
    if transformers is not None:
        input0 = copy.deepcopy(input)
        input0.preprocess(X=transformers[0],y=transformers[1])
    else:
        input0 = input

    inputs = SmartInput(input0)
    model = Sequential()
    bottom = inception_v3.InceptionV3(input_tensor=inputs,weights=None,include_top=False,pooling="avg")
    model.add(bottom)
    if parameters is not None:
        if "hidden" in parameters:
            for hidden in parameters["hidden"]:
                model.add(Dense(hidden))
                if "activation" in parameters:
                    model.add(Activation(parameters["activation"]))
                if "dropout" in parameters:
                    model.add(Dropout(parameters["dropout"]))
    model.add(SmartClassificationRegressionOutputSequential(input0))
    return model



def MobileNet(input=None,transformers=None,parameters=None):

    #Temporary transform data
    if transformers is not None:
        input0 = copy.deepcopy(input)
        input0.preprocess(X=transformers[0],y=transformers[1])
    else:
        input0 = input

    inputs = SmartInput(input0)
    model = Sequential()
    bottom = mobilenet.MobileNet(input_tensor=inputs,weights=None,include_top=False,pooling="avg")
    model.add(bottom)

    alpha = 1.0
    dropout = 1e-3
    if len(input0.y.shape)==1:
        raise Exception("ezmodel.eznetwork.MobileNet(): 'y' must be transformed into to categorical to work with Mobilenet!")
    else :
        classes = input0.y.shape[1]
    shape = (1, 1, int(1024 * alpha))

    model.add(Reshape(shape))
    model.add(Dropout(dropout, name='dropout'))
    model.add(Conv2D(classes, (1, 1),padding='same',name='conv_preds'))
    model.add(Activation('softmax', name='act_softmax'))
    model.add(Reshape((classes,)))

    return model

def MobileNetV2(input=None,transformers=None,parameters=None):

    #Temporary transform data
    if transformers is not None:
        input0 = copy.deepcopy(input)
        input0.preprocess(X=transformers[0],y=transformers[1])
    else:
        input0 = input

    inputs = SmartInput(input0)
    model = Sequential()
    bottom = mobilenet_v2.MobileNetV2(input_tensor=inputs,weights=None,include_top=False,pooling="avg")
    model.add(bottom)

    if parameters is not None:
        if "hidden" in parameters:
            for hidden in parameters["hidden"]:
                model.add(Dense(hidden))
                if "activation" in parameters:
                    model.add(Activation(parameters["activation"]))
                if "dropout" in parameters:
                    model.add(Dropout(parameters["dropout"]))
    model.add(SmartClassificationRegressionOutputSequential(input0))


    return model


def Xception(input=None,transformers=None,parameters=None):

    #Temporary transform data
    if transformers is not None:
        input0 = copy.deepcopy(input)
        input0.preprocess(X=transformers[0],y=transformers[1])
    else:
        input0 = input

    inputs = SmartInput(input0)
    model = Sequential()
    bottom = xception.Xception(input_tensor=inputs,weights=None,include_top=False,pooling="avg")
    model.add(bottom)
    if parameters is not None:
        if "hidden" in parameters:
            for hidden in parameters["hidden"]:
                model.add(Dense(hidden))
                if "activation" in parameters:
                    model.add(Activation(parameters["activation"]))
                if "dropout" in parameters:
                    model.add(Dropout(parameters["dropout"]))
    model.add(SmartClassificationRegressionOutputSequential(input0))

    return model



def VGG16(input=None,transformers=None,parameters=None):

    #Temporary transform data
    if transformers is not None:
        input0 = copy.deepcopy(input)
        input0.preprocess(X=transformers[0],y=transformers[1])
    else:
        input0 = input

    inputs = SmartInput(input0)
    model = Sequential()
    bottom = vgg16.VGG16(input_tensor=inputs,weights=None,include_top=False,pooling="avg")
    model.add(bottom)
    if parameters is not None:
        if "hidden" in parameters:
            for hidden in parameters["hidden"]:
                model.add(Dense(hidden))
                if "activation" in parameters:
                    model.add(Activation(parameters["activation"]))
                if "dropout" in parameters:
                    model.add(Dropout(parameters["dropout"]))
    else:
        model.add(Dense(4096,activation="relu"))
        model.add(Dense(4096,activation="relu"))
    model.add(SmartClassificationRegressionOutputSequential(input0))

    return model


def VGG19(input=None,transformers=None,parameters=None):
    #Temporary transform data
    if transformers is not None:
        input0 = copy.deepcopy(input)
        input0.preprocess(X=transformers[0],y=transformers[1])
    else:
        input0 = input

    inputs = SmartInput(input0)
    model = Sequential()
    bottom = vgg19.VGG19(input_tensor=inputs,weights=None,include_top=False,pooling="avg")
    model.add(bottom)
    if parameters is not None:
        if "hidden" in parameters:
            for hidden in parameters["hidden"]:
                model.add(Dense(hidden))
                if "activation" in parameters:
                    model.add(Activation(parameters["activation"]))
                if "dropout" in parameters:
                    model.add(Dropout(parameters["dropout"]))
    else:
        model.add(Dense(4096,activation="relu"))
        model.add(Dense(4096,activation="relu"))
    model.add(SmartClassificationRegressionOutputSequential(input0))

    return model


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization( name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(input=None,transformers=None,parameters=None):

    #Temporary transform data
    if transformers is not None:
        input0 = copy.deepcopy(input)
        input0.preprocess(X=transformers[0],y=transformers[1])
    else:
        input0 = input

    inputs = SmartInput(input0)

    x = ZeroPadding2D((3, 3))(inputs)
    print(x.get_shape())
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    print(x.get_shape())
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    print(x.get_shape())
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    print(x.get_shape())
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    print(x.get_shape())
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    #print(x.get_shape())
    #x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = GlobalAveragePooling2D()(x)

    outputs = SmartClassificationRegressionOutput(input0,x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# UNET definition using ezblocks4 to take into account depth
def UNET2(input=None,transformers=None,parameters=None):

    n = 64
    depth=4
    if parameters is not None:
        if "n" in parameters:
            n = parameters["n"]
        if "depth" in parameters:
            depth=parameters["depth"]

    conv = Block().define(
        """
        Conv2D(filters=?,kernel_size=(3,3),activation="relu",padding="same")
        Conv2D(filters=?,kernel_size=(3,3),activation="relu",padding="same")
        """
    )

    inputs = SmartInput(input)

    x = inputs

    #Encoder
    convs=[]
    for i in range(depth):
        x = conv(filters=n*(2**i)) (x)
        convs.append(x)
        x = MaxPooling2D(pool_size=(2,2)) (x)

    #bottleneck
    x = conv(filters=n*(2**(i+1))) (x)

    #Decoder
    for i in reversed(range(depth)):
        x = Conv2DTranspose(n*(2**i), (2, 2), strides=(2, 2), padding='same')(x)
        x = concatenate([x, convs[i]], axis=3)
        x = conv(filters=n*(2**i))(x)

    #Output
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    return model
