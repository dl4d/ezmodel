import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split,show_images
from ezmodel.ezblocks import *
from keras.applications.vgg16 import VGG16

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

input = Input(shape=train.X.shape[1:])
x = Conv2D(filters=4,kernel_size=(3,3),padding="same",activation="relu") (input)
x = MaxPooling2D(pool_size=(2,2)) (x)
x = Conv2D(filters=8,kernel_size=(3,3),padding="same",activation="relu") (x)
x = MaxPooling2D(pool_size=(2,2)) (x)
x = Conv2DTranspose(filters=8,kernel_size=(3,3),padding="same",activation="relu") (x)
x = UpSampling2D(size=(2,2))(x)
x = Conv2DTranspose(filters=4,kernel_size=(3,3),padding="same",activation="relu") (x)
x = UpSampling2D(size=(2,2))(x)
output_cae = Conv2D(filters=3,kernel_size=(1,1),padding="same",name="cae") (x)
#cae = Model(inputs=input,output=output)
#cae.summary()
vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=train.X.shape[1:], pooling="avg")
x = (vgg16) (output_cae)
x = Dense(100,activation="relu") (x)
output_global = Dense(7,activation="softmax",name="classification") (x)
model = Model(inputs = input, outputs=[output_cae,output_global])
model.summary()

# optimizer
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-4),
     "loss"     : {'classification': 'categorical_crossentropy', 'cae': 'mean_squared_error'},
     "metrics"  : {'classification': 'accuracy'}
}


model.compile(**optimizer)

#Temporary transform data
import copy
train0 = copy.deepcopy(train)
test0  = copy.deepcopy(test)
train0.preprocess(X=transformers[0],y=transformers[1])
test0.preprocess(X=transformers[0],y=transformers[1])



model.fit(train0.X,
          {'classification': train0.y, 'cae': train0.X},
          epochs=5,
          validation_data= (test0.X, {'classification': test0.y, 'cae': test0.X}),
          verbose=1)



# Training --------------------------------------------------------------------
parameters = {
    "epochs" : 50,
}
ez.train(parameters)




# model = keras.models.Sequential()
#
# a1  = ConvBlock(filters=32,kernel_size=(3,3),pooling=(2,2),padding="same")
# a2  = ConvBlock(filters=64,kernel_size=(3,3),pooling=(2,2),padding="same")
# a3  = ConvBlock(filters=128,kernel_size=(3,3),pooling=(2,2),padding="same")
# b1  = DeconvBlock(filters=128,kernel_size=(3,3),pooling=(2,2),padding="same")
# b2  = DeconvBlock(filters=64,kernel_size=(3,3),pooling=(2,2),padding="same")
# b3  = DeconvBlock(filters=32,kernel_size=(3,3),pooling=(2,2),padding="same")
# cae = ConnectCAE(input=train,transformers=transformers,blocks=[a1,a2,a3,b1,b2,b3])
# model.add(cae)
#
# pretrained = PretrainedBlock(path="vgg16",include_top=False,frozen=False,pooling="avg")
# model.add(pretrained.get(input_shape=train.X.shape[1:])[0])
#
# model.add(Dense(100,activation="relu"))
# model.add(Dense(7,activation="softmax"))
#
#
# model.summary()

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
