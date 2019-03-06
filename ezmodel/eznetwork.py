from keras.models import *
from keras.layers import *


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


#LeNet5
def LeNet5(input):

    #Checkers:
    if len(input.X.shape) !=4:
        raise Exception("\n\n \t [Fail] eznetwork.MLP(): LeNet5 convnet is not designed to work with this kind of inputs ! Please use another Network architecture ! ")

    inputs = SmartInput(input)
    x = Conv2D(6, kernel_size = (5, 5), strides=(1,1), padding="valid",input_shape=(32, 32, 1)) (inputs)
    x = Activation("relu") (x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (x)
    x = Conv2D(16, kernel_size = (5, 5), strides=(1,1), padding="valid") (x)
    x = Activation("relu") (x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (x)
    x = Flatten() (x)
    x = Dense(120) (x)
    x = Dense(84) (x)
    outputs = SmartClassificationRegressionOutput(input,x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

#Basic Multilayer Perceptron
def MLP(input, parameters):

    #Checkers:
    if len(input.X.shape) !=2:
        raise Exception("\n\n \t [Fail] eznetwork.MLP(): Multi Layer Perceptron are not designed to work with this kind of inputs ! Please use another Network architecture ! ")

    if parameters is None:
        raise Exception("\n\n \t [Fail] eznetwork.MLP(): Please provide a parameters list to MLP ! ")


    if "hidden" not in parameters:
        raise Exception("\n\n\t [Fail] eznetwork.MLP(): MLP with No Layer ! Please provide at leat one layer")
    else:
        if type(parameters["hidden"]) is int:
            hidden = [parameters["hidden"]]
        elif type(parameters["hidden"]) is list:
            hidden = parameters["hidden"]
        else:
            raise Exception("\n\n\t [Fail] eznetwork.MLP() : Please provide hidden layer parameters as a python list !")


    inputs = SmartInput(input)
    x = Dense(hidden[0]) (inputs)
    if "activation" not in parameters:
        x = Activation(parameters["activation"]) (x)
    if "dropout" not in parameters:
        x = Dropout(parameters["dropout"]) (x)
    for layer in hidden[1:]:
        x = Dense(layer) (x)
        if "activation" not in parameters:
            x = Activation(parameters["activation"]) (x)
        if "dropout" not in parameters:
            x = Dropout(parameters["dropout"]) (x)
    outputs = SmartClassificationRegressionOutput(input,x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# UNET network for segmentation
def UNET(input,parameters):
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

    model = Model(inputs=[inputs], outputs=[conv10])
    return model
