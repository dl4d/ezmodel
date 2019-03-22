import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split,show_images
from ezmodel.ezblocks import *

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
# An example of Convolutoinal Autoencoder (CAE)
a1  = ConvBlock(filters=64,kernel_size=(3,3),pooling=(2,2),padding="same")
a2  = ConvBlock(filters=128,kernel_size=(3,3),pooling=(2,2),padding="same")
a3  = ConvBlock(filters=256,kernel_size=(3,3),pooling=(2,2),padding="same")
b1  = DeconvBlock(filters=256,kernel_size=(3,3),pooling=(2,2),padding="same")
b2  = DeconvBlock(filters=128,kernel_size=(3,3),pooling=(2,2),padding="same")
b3  = DeconvBlock(filters=64,kernel_size=(3,3),pooling=(2,2),padding="same")

net       = ConnectCAE(input=train,transformers=transformers,blocks=[a1,a2,a3,b1,b2,b3])
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
# Evaluation ------------------------------------------------------------------
ez.evaluate()

p = ez.predict()
r = show_images(p,4)

input0 = copy.deepcopy(test)
input0.preprocess(X=transformers[0],y=transformers[1])
show_images(input0.y,4,samples=r)
