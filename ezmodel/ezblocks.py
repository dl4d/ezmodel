from keras.models import *
from keras.layers import *
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

        #DenseBlock ----------------------------------------------------------
        if type(block) == DenseBlock:
            #Check need of flatten layers to maker the junction between conv and dense blocks
            if (type(previous_block) == ConvBlock) or (type(previous_block) == InceptionBlock):
                x = Flatten() (previous)
                previous = x
            x = Dense(block.units) (previous)
            if block.activation is not None:
                x = Activation(block.activation) (x)
            if block.dropout is not None:
                x = Dropout(block.dropout) (x)
            if block.bn is not None:
                x = BatchNormalization() (x)
            previous = x
            previous_block = block
        # --------------------------------------------------------------------

        #ConvBlock -----------------------------------------------------------
        if type(block) == ConvBlock:
            x = Conv2D(filters=block.filters,
                       kernel_size=block.kernel_size,
                       strides=block.strides,
                       padding=block.padding,
            ) (previous)
            if block.activation is not None:
                x = Activation(block.activation) (x)
            if block.dropout is not None:
                x = Dropout(block.dropout) (x)
            if block.pooling is not None:
                x = MaxPooling2D(pool_size=block.pooling) (x)
            if block.bn is not None:
                x = BatchNormalization() (x)
            previous = x
            previous_block = block
        # --------------------------------------------------------------------

        #InceptionBlock ------------------------------------------------------
        if type(block) == InceptionBlock:

            #tower1
            tower1 = Conv2D(filters=block.filters,kernel_size=(1,1),padding="same") (previous)
            if block.activation is not None:
                tower1 = Activation(block.activation) (tower1)
            tower1 = Conv2D(filters=block.filters,kernel_size=(3,3),padding="same") (tower1)
            if block.activation is not None:
                tower1 = Activation(block.activation) (tower1)

            #tower2
            tower2 = Conv2D(filters=block.filters,kernel_size=(1,1),padding="same") (previous)
            if block.activation is not None:
                tower2 = Activation(block.activation) (tower2)
            tower2 = Conv2D(filters=block.filters,kernel_size=(5,5),padding="same") (tower2)
            if block.activation is not None:
                tower2 = Activation(block.activation) (tower2)


            #tower3
            tower3 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(previous)
            tower3 = Conv2D(filters=block.filters,kernel_size=(1,1),padding="same") (tower3)
            if block.activation is not None:
                tower3 = Activation(block.activation) (tower3)

            x = concatenate([tower1,tower2,tower3],axis=3)
            previous = x
            previous_block = block



    outputs = SmartClassificationRegressionOutput(input0,x)
    model = Model(inputs=inputs,outputs=outputs)
    return model
