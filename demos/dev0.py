import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split,show_images
#from keras.layers import *
#from keras.models import Model
# from ezmodel.ezblocks2 import *
from ezmodel.ezblocks3 import *


import keras

# [EZSET]  -------------------------------------------------------------------
parameters = {
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\bacteria\\",
    "resize"      : (32,32)
}
data = ezset(parameters)

#Preprocessing
data.autoencoder()

#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Transform
transformers = train.transform(X="standard",y="standard")


# [EZNETWORK with custome EZBLOCKS]

conv = Block().define(
"""
    Conv2D(filters=?,kernel_size=?)
    MaxPooling2D(pool_size=?)
    BatchNormalization(momentum=?)
"""
)

dense = Block().define(
"""
    Dense(units=?)
    Dropout(rate=?)
"""
)

#inp(Input(shape=train.X))
graph =Graph(train.X)
graph("",conv(filters=128,kernel_size=(3,3),pool_size=(2,2),momentum=0.8))
graph("",conv(filters=128,kernel_size=(3,3),pool_size=(2,2),momentum=0.8))
# graph("_",GlobalAveragePooling2D())
# graph("",conv(filters=128,kernel_size=(3,3),pool_size=(2,2),momentum=0.8))
# graph("_",GlobalAveragePooling2D())
# graph("",Concatenate())
# graph("",Dense(120))
# graph("",Dense(80))

# print(graph.tuple)

sys.exit()

# ezn = eznet()
#
# ezn.input(data.X)
# ezn.add(conv(filters=128,kernel_size=(3,3),pool_size=(2,2),momentum=0.8))
# ezn.add(conv(filters=128,kernel_size=(3,3),pool_size=(2,2),momentum=0.8))
# ezn.add(GlobalAveragePooling2D())
# ezn.add(dense(units=120,rate=0.5))
# ezn.add(dense(units=80,rate=0.5))
# ezn.output(Dense(4,activation="softmax"))
# model = ezn.gen()




# dense(units=120,rate=0.5)
print(a)

sys.exit()



# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def new(s,*args,**kwargs):
    for k,v in kwargs.items():
        tofind=k+"=?"
        replace=k+"="+str(v)
        s = s.replace(tofind,replace)
    a = s.splitlines()
    block=[]
    for l in a:
        if len(l)==0:
            continue;
        block.append(eval(l))
    return block

def sequentially(data,blocks):
    input = Input(shape=data.X.shape[1:])
    x = input
    for block in blocks:
        if type(block) is not list:
            block = [block]
        for sub in block:
            x = (sub) (x)

    m = Model(inputs=input,outputs=x)
    return m

convblock = """
Conv2D(filters=?,kernel_size=?)
MaxPooling2D(pool_size=?)
"""

denseblock = """
Dense(units=?)
Dropout(rate=?)
"""

model = sequentially(train,[
new(convblock,filters=128,kernel_size=(3,3),pool_size=(2,2)),
new(convblock,filters=256,kernel_size=(3,3),pool_size=(2,2)),
GlobalAveragePooling2D(),
new(denseblock,units=120,rate=0.5),
new(denseblock,units=120,rate=0.5),
Dense(80)
]
)

model.summary()
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------








#
# print(newblock(s,128,(3,3)))
#
#
# s = s.replace("filters=?","filters=128")
# s = s.replace("kernel_size=?","kernel_size=(3,3)")
# a = s.splitlines()
# block=[]
# for l in a:
#     if len(l)==0:
#         continue;
#     block.append(eval(l))
#
# print(block)

# print(eval(s))

#
# block = [
#     Conv2D,["filters","kernel_size"],
#     MaxPooling2D,["pool_size"],
#     BatchNormalization,[]
# ]
# print(block)
#
# for layer in block:
#     if type(layer) is type:
#         print(layer)

# # Ca ca marche
# convblock = lambda **kwargs : (
#     Conv2D(**kwargs),
#     MaxPooling2D()
# )
# a = convblock(filters=128,kernel_size=(3,3))
# print(a[0].get_config()["filters"])
#
# convblock = lambda **kwargs : (
#     Conv2D(**kwargs),
#     MaxPooling2D()
# )
# a = convblock(filters=128,kernel_size=(3,3),pool_size=(2,2))
# print(a[0].get_config()["filters"])



# Ca ca marche
# convblock = lambda **kwargs : (
#     Conv2D(filters=kwargs["filters"],kernel_size=kwargs["kernel_size"]),
#     MaxPooling2D(pool_size=kwargs["pool_size"])
# )
# print(convblock(filters=128,kernel_size=(3,3),pool_size=(2,2)))



#print(convblock(128,(3,3)))


#
# b = newblock(
#         _Conv2D("filters","kernel_size"),
#         _MaxPooling2D()
#         )
#
#
# func_template = """
#                 def conv(filters=%d,kernel_size=(%d,%d)):
#                     return Conv2D(filters=%d,kernel_size=%d)
#                 """
# exec(func_template % (128,3,3))


# func_template = """def activate_field_%d(): print(%d)"""
# for x in range(1, 11): exec(func_template % (x, x))
#
# activate_field_1()




# def init(block,**kwargs):
#     for i in range(len(block)):
#         cur = block[i]
#         print("-->",cur)
#         print(len(cur))
#         func = cur[0]
#         if len(cur)>1:
#             args = cur[1]
#             print("args=",args)
#             print("args=",len(args))
#             for j in range(len(args)):
#                 if args[j] in kwargs:
#                     print('trouve')
#         else:
#             print(func())


# init(b,filter=128,kernel_size=(3,3),momentum=1)











# from collections import OrderedDict
#
# layers = [
#     (Conv2D, ["filters","kernel_size"]),
#     (MaxPooling2D, ["pool_size"])
# ]
# a = OrderedDict(layers)
# print(a)
#
# for k,v in a.items():
#     function = k
#     arg      = v
#     x = lambda v: function(v)
#     break
#
# print(x([128,(3,3)]))


# a = dict(a)
# print(a)
#
# for k,v in a.items():
#     print(k)
#     print(v)
#     print("\n")
#     m = lambda v : k()
#
# m("aa")










#
# net = eznet(input=train)
#
# conv = net.newblock(
#     Conv2D(filters="?",kernel_size=(3,3),padding="same",activation="relu"),
#     MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format=None),
#     BatchNormalization(momentum="?")
#     )
# deconv = net.newblock(
#     Conv2DTranspose(filters="?",kernel_size=(3,3),padding="same",activation="relu"),
#     UpSampling2D(size=(2,2))
#     )
# dense = net.newblock(
#     Dense(units="?")
# )
#
#
# net.add(
# conv(filters=16,momentum=0.8),
# conv(filters=32,momentum=0.8),
# deconv(filters=32),
# deconv(filters=16),
# keras.layers.GlobalAveragePooling2D(),
# dense(units=100)
# )
# # _   = conv(filters=16,momentum=0.8).add(net)
# # _   = conv(filters=32,momentum=0.8).add(net)
#
# m = net.gen()
# m.summary()

#
# print(net.blocks["block_0"].layers[0].get_config()["filters"])
# print(net.blocks["block_1"].layers[0].get_config()["filters"])

# x  = conv(filters=64)
# _1_x1 = conv(filters=128).linkto(x)
# _1_x2 = conv(filters=128).linkto(x)
# _  = merge(x1,x2)
# _  = Dense(100)




# c3 = conv(filters=128)
# d1 = deconv(filter=128)
# d2 = deconv(filter=64)
# d3 = deconv(filter=32)
# d4 = Conv2D(filters=3,kernel_size=(1,1),padding="same",name="cae")

# #Linkage
# input = Input(shape=train.X.shape[1:])
# a = x[0] (input)
#
# for layer in x[1:]:
#     a = layer (a)
#
# model = keras.models.Model(input=input,output=a)
# model.summary()



#
# input = Input(shape=train.X.shape[1:])
# x = Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu") (input)
# x = MaxPooling2D(pool_size=(2,2)) (x)
# x = Conv2D(filters=128,kernel_size=(3,3),padding="same",activation="relu") (x)
# x = MaxPooling2D(pool_size=(2,2)) (x)
# x = Conv2D(filters=256,kernel_size=(3,3),padding="same",activation="relu") (x)
# x = MaxPooling2D(pool_size=(2,2)) (x)
# x = Conv2DTranspose(filters=256,kernel_size=(3,3),padding="same",activation="relu") (x)
# x = UpSampling2D(size=(2,2))(x)
# x = Conv2DTranspose(filters=128,kernel_size=(3,3),padding="same",activation="relu") (x)
# x = UpSampling2D(size=(2,2))(x)
# x = Conv2DTranspose(filters=64,kernel_size=(3,3),padding="same",activation="relu") (x)
# x = UpSampling2D(size=(2,2))(x)
# output_cae = Conv2D(filters=3,kernel_size=(1,1),padding="same",name="cae") (x)
#







#
# dense = net.Blocks(
#     Dense("?",activation="relu")
#     )
#
# x       = convpoolbn(64,(3,3))
# x       = convpoolbn(filters=128,kernel_size=(3,3))
# x1      = convpoolbn(anc=x,filters=256,kernel_size=(3,3))
# x2      = conv(anc=x,filters=256,kernel_size=(3,3))
# merge   = Merge(x1,x2)
# dense = dense(anc=merge,units=120)
#
#














# An example of Convolutoinal Autoencoder (CAE)
#Encoder
# a1  = ConvBlock(filters=64,kernel_size=(3,3),pooling=(2,2),padding="same")
# a2  = ConvBlock(filters=128,kernel_size=(3,3),pooling=(2,2),padding="same")
# a3  = ConvBlock(filters=256,kernel_size=(3,3),pooling=(2,2),padding="same")
# b1  = DeconvBlock(filters=256,kernel_size=(3,3),pooling=(2,2),padding="same")
# b2  = DeconvBlock(filters=128,kernel_size=(3,3),pooling=(2,2),padding="same")
# b3  = DeconvBlock(filters=64,kernel_size=(3,3),pooling=(2,2),padding="same")
#
# net       = ConnectCAE(input=train,transformers=transformers,blocks=[a1,a2,a3,b1,b2,b3])
# net.summary()
#
# # [Keras Optimizer, Loss & Metrics]  ------------------------------------------
# optimizer = {
#     "optimizer" : keras.optimizers.Adam(lr=1e-4),
#     "loss"      : keras.losses.mean_squared_error,
#     "metrics"   : [keras.metrics.mean_squared_error]
# }
# # [EZMODEL]  ------------------------------------------------------------------
# ez = ezmodel(
#     train = train,
#     test  = test,
#     network = net,
#     optimizer = optimizer,
#     transformers = transformers
# )
# # Training --------------------------------------------------------------------
# parameters = {
#     "epochs" : 50,
#     # "validation_split": 0.2
# }
# ez.train(parameters)
# # Evaluation ------------------------------------------------------------------
# ez.evaluate()
#
# p = ez.predict()
# r = show_images(p,4)
#
# input0 = copy.deepcopy(test)
# input0.preprocess(X=transformers[0],y=transformers[1])
# show_images(input0.y,4,samples=r)
