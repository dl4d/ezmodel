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
parameters={
    # "name"      : "Mitochondria",
    "path"      : "C:\\Users\\daian\\Desktop\\DATA\\Mito\\images\\",
    "path_mask" : "C:\\Users\\daian\\Desktop\\DATA\\Mito\\masks\\",
    "resize"    : (128,128)
}
data = ezset(parameters)
data.autoencoder()
#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Transform
transformers = train.transform(X="standard",y="minmax")

# [EZNETWORK with custome EZBLOCKS]
# An example of Convolutoinal Autoencoder (CAE)
#Encoder
inp = Input(shape=train.X.shape[1:])
a  = ConvBlock(filters=64,kernel_size=(3,3),pooling=(2,2),padding="same")
x = a.get()[0] (inp)
for layer in a.get()[1:]:
    if layer is None:
        continue
    x = layer (x)
b  = ConvBlock(filters=128,kernel_size=(3,3),pooling=(2,2),padding="same")
for layer in b.get():
    if layer is None:
        continue
    x = layer (x)
c  = ConvBlock(filters=256,kernel_size=(3,3),pooling=(2,2),padding="same")
for layer in c.get():
    if layer is None:
        continue
    x = layer (x)

x = Flatten() (x)
# x = Dense(1024,activation="relu") (x)
x  = Dense(2,activation="linear") (x)
encoder = Model(input=inp,output=x)
encoder.summary()

inp2 = Input(shape=encoder.output_shape[1:])

x = Dense(16*16*256) (inp2)
x = Reshape((16,16,256)) (x)

d  = DeconvBlock(filters=256,kernel_size=(3,3),pooling=(2,2),padding="same")
# x = d.get()[0] (inp2)
# for layer in d.get()[1:]:
for layer in d.get():
    if layer is None:
        continue
    x = layer (x)
e  = DeconvBlock(filters=128,kernel_size=(3,3),pooling=(2,2),padding="same")
for layer in e.get():
    if layer is None:
        continue
    x = layer (x)
f  = DeconvBlock(filters=64,kernel_size=(3,3),pooling=(2,2),padding="same")
for layer in f.get():
    if layer is None:
        continue
    x = layer (x)

x = Conv2D(filters=1,kernel_size=(1,1),activation="linear") (x)

decoder = Model(input = inp2, output = x)

net = Sequential()
net.add(encoder)
net.add(decoder)

#net       = ConnectCAE(input=train,transformers=transformers,blocks=[a1,a2,a3,p1,b1,b2,b3])
net.summary()


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


p = ez.predict()
r = show_images(p,4)

input0 = copy.deepcopy(test)
input0.preprocess(X=transformers[0],y=transformers[1])
show_images(input0.y,4,samples=r)

print(p.max())
print(p.min())

a = encoder.predict(input0.X)
print(a.shape)

import matplotlib.pyplot as plt
print(a)
plt.scatter(a[:,0],a[:,1])
plt.show()
