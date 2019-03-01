from sklearn.model_selection import train_test_split
from keras.models import Input,Model
from keras.layers import *
from keras import layers
import keras
import keras.backend as K


class eztrainer:

    def __init__(self):

        self.network = None
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None

    def gen_trainval(self,ezdata,size=0.2,random_state=42):

        self.X_train,self.X_valid,self.y_train,self.y_valid = train_test_split(ezdata.X,ezdata.y,test_size=size,random_state=42)

        print ("[X] Train/Validation set generation (size = ",str(size),"): Done")
        print ("--- Training set : ", self.X_train.shape[0], "images")
        print ("--- Validation set     : ", self.X_valid.shape[0], "images")
        print("\n")

    def gen_network(self,inputs,outputs):
        m = Model(inputs=inputs,outputs=outputs)
        self.network = m

    def Input(self):
        return Input(shape=self.X_train.shape[1:])

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

    def Network(self,name=None,parameters=None):
        if name is None:
            raise Exception("[Fail] eztrainer.Network(): Please enter a name for a prebuilt neural network")

        if name.lower() == "unet":
            self.network = self.Network_UNET(parameters)


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
