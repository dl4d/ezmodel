from keras.models import *
from keras.layers import *
from keras.applications import *
from ezmodel.eznetwork import SmartInput,SmartClassificationRegressionOutput
import copy

class DenseBlock:

    def __init__(self,units,activation=None,dropout=None,bn=None):


        if units is None:
            raise Exception("ezmodel.DenseBlock() : Please define a number of units (neuron) into parameters")

        self.units = units
        self.activation = activation
        self.dropout = dropout
        self.bn = bn
        self.father = None

    def __call__(self,object):
        self.father = object

    def get(self,inputs):

        x = Dense(self.units) (inputs)
        if self.activation is not None:
            x = Activation(self.activation) (x)
        if self.dropout is not None:
            x = Dropout(self.dropout) (x)
        if self.bn is not None:
            x = BatchNormalization() (x)
        return x

    # def copy(self,units=None,activation=None,dropout=None,bn=None):
    #
    #     a = copy.deepcopy(self)
    #
    #     if units is not None:
    #         a.units = units
    #     if activation is not None:
    #         a.activation = activation
    #     if dropout is not None:
    #         a.dropout = dropout
    #     if bn is not None:
    #         a.bn=bn
    #
    #     return a

class ConvBlock:

    def __init__(self,filters,kernel_size,strides=(1,1),padding="valid",activation="relu",dropout=None,pooling=(2,2),bn=None):

        if filters is None:
            raise Exception("ezmodel.ConvBlock() : Please define a number of filters with 'filters' parameters")

        if kernel_size is None:
            raise Exception("ezmodel.ConvBlock() : Please define a kernel size with 'kernel_size' parameters")


        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.dropout = dropout
        self.pooling = pooling
        self.bn = bn
        self.father = None

    def __call__(self,object):
        self.father = object

    def get(self,inputs):
        x = Conv2D(filters=self.filters,kernel_size=self.kernel_size,strides=self.strides,padding=self.padding) (inputs)
        if self.activation is not None:
            x = Activation(self.activation) (x)
        if self.dropout is not None:
            x = Dropout(self.dropout) (x)
        if self.pooling is not None:
            x = MaxPooling2D(pool_size=self.pooling) (x)
        if self.bn is not None:
            x = BatchNormalization() (x)
        return x


    # def copy(self,filters=None,kernel_size=None,strides=None,padding=None,activation=None,dropout=None,bn=None):
    #
    #     a = copy.deepcopy(self)
    #
    #     if filters is not None:
    #         a.filters = filters
    #     if kernel_size is not None:
    #         a.kernel_size = kernel_size
    #     if strides is not None:
    #         a.strides = strides
    #     if padding is not None:
    #         a.padding = padding
    #     if activation is not None:
    #         a.activation = activation
    #     if dropout is not None:
    #         a.dropout = dropout
    #     if bn is not None:
    #         a.bn=bn
    #
    #     return a



class InceptionBlock:

    def __init__(self,filters,activation="relu"):

        if filters is None:
            raise Exception("ezmodel.InceptionBlock() : Please define a number of filters with 'filters' parameters")

        self.filters = filters
        self.activation = activation
        self.father = None

    def __call__(self,object):
        self.father = object

    def get(self,inputs):
        #tower1
        tower1 = Conv2D(filters=self.filters,kernel_size=(1,1),activation=self.activation,padding="same") (inputs)
        tower1 = Conv2D(filters=self.filters,kernel_size=(3,3),activation=self.activation,padding="same") (tower1)
        #tower2
        tower2 = Conv2D(filters=self.filters,kernel_size=(1,1),activation=self.activation,padding="same") (inputs)
        tower2 = Conv2D(filters=self.filters,kernel_size=(5,5),activation=self.activation,padding="same") (tower2)
        #tower3
        tower3 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(inputs)
        tower3 = Conv2D(filters=self.filters,kernel_size=(1,1),activation=self.activation,padding="same") (tower3)

        return concatenate([tower1,tower2,tower3],axis=3)

class NaiveInceptionBlock:
    def __init__(self,filters,activation="relu"):

        if filters is None:
            raise Exception("ezmodel.InceptionBlock() : Please define a number of filters with 'filters' parameters")

        self.filters = filters
        self.activation = activation
        self.father = None

    def __call__(self,object):
        self.father = object

    def get(self,inputs):
        tower1 = Conv2D(filters=self.filters,kernel_size=(1,1),padding="same") (inputs)
        tower2 = Conv2D(filters=self.filters,kernel_size=(3,3),padding="same") (inputs)
        tower3 = Conv2D(filters=self.filters,kernel_size=(5,5),padding="same") (inputs)
        tower4 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(inputs)
        return concatenate([tower1,tower2,tower3,tower4],axis=3)

class InceptionBlockDimReduce:
    def __init__(self,filters,activation="relu"):

        if filters is None:
            raise Exception("ezmodel.InceptionBlock() : Please define a number of filters with 'filters' parameters")

        self.filters = filters
        self.activation = activation
        self.father = None
    def __call__(self,object):
        self.father = object

    def get(self,inputs):
        tower1 = Conv2D(filters=self.filters,kernel_size=(1,1),padding="same") (inputs)

        tower2 = Conv2D(filters=self.filters,kernel_size=(1,1),padding="same") (inputs)
        tower2 = Conv2D(filters=self.filters,kernel_size=(3,3),padding="same") (tower2)

        tower3 = Conv2D(filters=self.filters,kernel_size=(1,1),padding="same") (inputs)
        tower3 = Conv2D(filters=self.filters,kernel_size=(5,5),padding="same") (tower3)

        tower4 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(inputs)
        tower4 = Conv2D(filters=self.filters,kernel_size=(1,1),padding="same") (tower4)

        return concatenate([tower1,tower2,tower3,tower4],axis=3)


class PretrainedBlock:

    def __init__(self,path,include_top=False,frozen=True,pooling="avg"):

        if path is None:
            raise Exception("ezmodel.PretrainedBlock() : Please define a model name or path")

        self.path = path
        self.include_top = include_top
        self.frozen = frozen
        self.pooling = pooling
        self.father = None


    def __call__(self,object):
        self.father = object

    def get(self,inputs):
        if self.path.lower()=="mobilenet":
            pretrained = mobilenet.MobileNet(include_top=self.include_top, weights='imagenet', input_shape=inputs.get_shape().as_list()[1:],pooling=self.pooling)
        elif self.path.lower()=="mobilenetv2":
            pretrained = mobilenet_v2.MobileNetV2(include_top=self.include_top, weights='imagenet', input_shape=inputs.get_shape().as_list()[1:],pooling=self.pooling)
        elif self.path.lower()=="vgg16":
            pretrained = vgg16.VGG16(include_top=self.include_top, weights='imagenet', input_shape=inputs.get_shape().as_list()[1:], pooling=self.pooling)
        elif self.path.lower()=="vgg19":
            pretrained = vgg19.VGG19(include_top=self.include_top, weights='imagenet', input_shape=inputs.get_shape().as_list()[1:],pooling=self.pooling)
        elif self.path.lower()=="xception":
            pretrained = xception.Xception(include_top=self.include_top, weights='imagenet', input_shape=inputs.get_shape().as_list()[1:], pooling=self.pooling)
        elif self.path.lower()=="inceptionv3":
            pretrained = inception_v3.InceptionV3(include_top=self.include_top, weights='imagenet', input_shape=inputs.get_shape().as_list()[1:], pooling=self.pooling)
        else:
            #check local model
            if os.path.isfile(self.path+".h5"):
                pretrained = load_model(self.path+".h5")
            else:
                raise Exception("eznetwork.Pretrained(): No model path: ",self.path," has been found !")

        if self.frozen:
            for layer in pretrained.layers:
                    layer.trainable = False

        return pretrained


# def ConnectBlock2(input=None,transformers=None,blocks=None):
#     """
#     Connect ezblock sequentially
#     TODO : Need a way to connect block functionnally, may be define a ConnectBlockFunctionnaly ?
#     """
#     #Temporary transform data
#     if transformers is not None:
#         input0 = copy.deepcopy(input)
#         input0.preprocess(X=transformers[0],y=transformers[1])
#     else:
#         input0 = input
#
#     model = Sequential()
#
#     for block in blocks:
#         model.add(block.get())
#     model.add(SmartClassificationRegressionOutputSequential(input0))


def ConnectBlock(input=None,transformers=None,blocks=None):

    """
    Connect ezblock sequentially
    TODO : Need a way to connect block functionnally, may be define a ConnectBlockFunctionnaly ?
    """

    #Temporary transform data
    if transformers is not None:
        input0 = copy.deepcopy(input)
        input0.preprocess(X=transformers[0],y=transformers[1])
    else:
        input0 = input


    inputs = SmartInput(input0)

    previous = inputs
    previous_block = None
    for block in blocks:


        if type(block) == DenseBlock:
            #Check need of flatten layers to maker the junction between conv and dense blocks
            if (type(previous_block) == ConvBlock) or (type(previous_block) == InceptionBlock) or (type(previous_block) == NaiveInceptionBlock) or (type(previous_block) == InceptionBlockDimReduce):
                x = Flatten() (previous)
                previous = x
            # -----------------------------------------------------------------

        x = block.get(previous)
        previous = x
        previous_block = block


    outputs = SmartClassificationRegressionOutput(input0,x)
    model = Model(inputs=inputs,outputs=outputs)
    return model
