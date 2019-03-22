import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split,show_images
from ezmodel.ezblocks import *

import keras

# [EZSET]  -------------------------------------------------------------------

# [EZSET]
parameters = {
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\Skin\\skin.npz",
    "X.key"       : "images",
    "y.key"       : "labels",
    "synsets.key" : "synsets"
}
data = ezset(parameters)

#Preprocessing
#data.autoencoder()

#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Transform
transformers = train.transform(X="minmax",y="categorical")

print(train.X.shape)
print(train.y.shape)

# [EZNETWORK with custome EZBLOCKS]

model = keras.models.Sequential()

a1  = ConvBlock(filters=32,kernel_size=(3,3),pooling=(2,2),padding="same")
a2  = ConvBlock(filters=64,kernel_size=(3,3),pooling=(2,2),padding="same")
a3  = ConvBlock(filters=128,kernel_size=(3,3),pooling=(2,2),padding="same")
b1  = DeconvBlock(filters=128,kernel_size=(3,3),pooling=(2,2),padding="same")
b2  = DeconvBlock(filters=64,kernel_size=(3,3),pooling=(2,2),padding="same")
b3  = DeconvBlock(filters=32,kernel_size=(3,3),pooling=(2,2),padding="same")
cae = ConnectCAE(input=train,transformers=transformers,blocks=[a1,a2,a3,b1,b2,b3])
model.add(cae)

pretrained = PretrainedBlock(path="vgg16",include_top=False,frozen=False,pooling="avg")
model.add(pretrained.get(input_shape=train.X.shape[1:])[0])

model.add(Dense(100,activation="relu"))
model.add(Dense(7,activation="softmax"))


model.summary()

optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-4),
    "loss"      : keras.losses.categorical_crossentropy,
    "metrics"   : [keras.metrics.categorical_crossentropy]
}

model.compile(**optimizer)

# [EZMODEL]  ------------------------------------------------------------------
ez = ezmodel(
    train = train,
    test  = test,
    network = model,
    optimizer = optimizer,
    transformers = transformers
)
# Training --------------------------------------------------------------------
parameters = {
    "epochs" : 50,
    # "validation_split": 0.2
}
ez.train(parameters)

# inp = Input(shape=train.X.shape[1:])
#
# # An example of Convolutoinal Autoencoder (CAE)
# a1  = ConvBlock(filters=32,kernel_size=(3,3),pooling=(2,2),padding="same")
# a2  = ConvBlock(filters=64,kernel_size=(3,3),pooling=(2,2),padding="same")
# a3  = ConvBlock(filters=128,kernel_size=(3,3),pooling=(2,2),padding="same")
# b1  = DeconvBlock(filters=128,kernel_size=(3,3),pooling=(2,2),padding="same")
# b2  = DeconvBlock(filters=64,kernel_size=(3,3),pooling=(2,2),padding="same")
# b3  = DeconvBlock(filters=32,kernel_size=(3,3),pooling=(2,2),padding="same")
# cae = ConnectCAE(input=train,transformers=transformers,blocks=[a1,a2,a3,b1,b2,b3])
# cae.name = "cae"
# #cae.summary()
#
# #VGG16
# model = keras.models.Sequential()
# pretrained = PretrainedBlock(path="vgg16",include_top=False,frozen=False,pooling="avg")
# model.add(pretrained.get(input_shape=train.X.shape[1:])[0])
# model.add(Dense(100,activation="relu"))
# model.add(Dense(7,activation="softmax"))
# model.name = "vgg16"
#
# x = Multiply()[cae(inp),model(inp)]
#model = keras.models.Sequential()
# model.add(cae)
# model.add(pretrained.get(input_shape=train.X.shape[1:])[0])
# model.summary()
# inp = Input(shape=train.X.shape[1:])
# m0 = PretrainedBlock(path="mobilenet",include_top=False,frozen=False,pooling="avg").get(input_shape=train.X.shape[1:])[0]
# m0.name='MobileNet'
# m1 = PretrainedBlock(path="mobilenet",include_top=False,frozen=True,pooling="avg").get(input_shape=train.X.shape[1:])[0]
# m1.name='FrozenMobileNet'
# x = Multiply()([m0(inp), m1(inp)])
# out = Dense(2, activation='softmax')(x)
# model = Model(inputs=inp, outputs=out)
# model.summary()


# [Keras Optimizer, Loss & Metrics]  ------------------------------------------
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-4),
    "loss"      : keras.losses.mean_squared_error,
    "metrics"   : [keras.metrics.mean_squared_error]
}
# [EZMODEL]  ------------------------------------------------------------------
ez = ezmodel(
    train = train,
    test  = test,
    network = net,
    optimizer = optimizer,
    transformers = transformers
)
# Training --------------------------------------------------------------------
parameters = {
    "epochs" : 50,
    # "validation_split": 0.2
}
ez.train(parameters)
# Evaluation ------------------------------------------------------------------
ez.evaluate()

p = ez.predict()
r = show_images(p,4)

input0 = copy.deepcopy(test)
input0.preprocess(X=transformers[0],y=transformers[1])
show_images(input0.y,4,samples=r)
