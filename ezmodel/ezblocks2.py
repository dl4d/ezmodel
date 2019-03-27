from keras.models import *
from keras.layers import *
from keras.applications import *
import copy
import os
import sys
import keras


class eznet:

    def __init__(self,input=None,transformers=None):
        print("Eznet initialization")
        self.input = input
        self.input_tensor = None
        self.graph  = None

    def newblock(self,*args):
        blocks=[]
        args_s=[]
        kwargs_s=[]
        for layer in args:
            blocks.append(layer[0])
            args_s.append(layer[1])
            kwargs_s.append(layer[2])
        return Block(blocks,args_s,kwargs_s)

    def add(self,*args):

        if self.graph is None:
            input = Input(shape=self.input.X.shape[1:])
            self.input_tensor = input

        for cur in args:
            if type(cur) is Block:
                x = cur.layers[0](input)
                for layer in cur.layers[1:]:
                    x = (layer) (x)
            else:
                x = (cur) (x)

            input=x

        self.graph = x

        # cur = copy.deepcopy(self)
        # if net.graph is None:
        #     input = Input(shape=net.input.X.shape[1:])
        #     net.input_tensor = input
        # else:
        #     input = net.graph
        #
        # x = cur.layers[0] (input)
        # for layer in cur.layers[1:]:
        #     x = (layer) (x)
        #
        # net.graph = x
        # return x


    def gen(self):
        return Model(input=self.input_tensor,output=self.graph)

# Block -----------------------------------------------------------------------
class Block:

    def __init__(self,blocks,args_s,kwargs_s):
        self.name   = None
        self.blocks = blocks
        self.args   = args_s
        self.kwargs = kwargs_s
        self.layers = None

    def __call__(self,*args,**kwargs):

        cur = copy.deepcopy(self)

        if "name" in kwargs:
            cur.name = kwargs["name"]

        layers =[]
        for layer in range(len(cur.blocks)):
            for key, value in cur.kwargs[layer].items():
                if value=="?":
                    cur.kwargs[layer][key] = kwargs[key]

            layers.append(cur.blocks[layer](**cur.kwargs[layer]))
        cur.layers = layers

        return cur

    #
    # def add(self,net,ancestor=None):
    #
    #     cur = copy.deepcopy(self)
    #     if net.graph is None:
    #         input = Input(shape=net.input.X.shape[1:])
    #         net.input_tensor = input
    #     else:
    #         input = net.graph
    #
    #     x = cur.layers[0] (input)
    #     for layer in cur.layers[1:]:
    #         x = (layer) (x)
    #
    #     net.graph = x
    #     return x



    # def add(self,net,ancestor=None):
    #
    #     cur = copy.deepcopy(self)
    #
    #     id = len(net.graph)
    #     cur.name = "block_" + str(id)
    #
    #     if ancestor is None:
    #         anc = id-1
    #     else:
    #         for i in range(len(net.blocks)):
    #             if net.blocks[i] is ancestor:
    #                 anc = i
    #
    #     net.graph.append((id,anc))
    #     net.blocks.append(cur)
    #
    #
    #     print("Adding ",cur.name," to network graph.")
    #     return cur



# -----------------------------------------------------------------------------


# Keras layer redefinition
def Conv2D(*args,**kwargs):
    from keras.layers import Conv2D as k_Conv2D
    return (k_Conv2D,args,kwargs)

def Conv2DTranspose(*args,**kwargs):
    from keras.layers import Conv2DTranspose as k_Conv2DTranspose
    return (k_Conv2DTranspose,args,kwargs)

def MaxPooling2D(*args,**kwargs):
    from keras.layers import MaxPooling2D as k_MaxPooling2D
    return (k_MaxPooling2D,args,kwargs)

def UpSampling2D(*args,**kwargs):
    from keras.layers import UpSampling2D as k_Upsampling2D
    return (k_Upsampling2D,args,kwargs)

def Activation(*args,**kwargs):
    from keras.layers import Activation as k_Activation
    return (k_Activation,args,kwargs)

def BatchNormalization(*args,**kwargs):
    from keras.layers import BatchNormalization as k_BatchNormalization
    return (k_BatchNormalization,args,kwargs)

def Dense(*args,**kwargs):
    from keras.layers import Dense as k_Dense
    return (k_Dense,args,kwargs)
#
# def GlobalAveragePooling2D(*args,**kwargs):
#     from keras.layers import GlobalAveragePooling2D as k_GlobalAveragePooling2D
#     return (k_GlobalAveragePooling2D,args,kwargs)
