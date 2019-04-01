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
from ezmodel.ezblocks4 import *

conv = Block().new(
"""
    Conv2D(filters=?,kernel_size=(3,3),padding="same")
    MaxPooling2D(pool_size=(2,2))
    BatchNormalization(momentum=0.8)
"""
)

deconv = Block().new(
"""
    UpSampling2D(size=(2,2))
    Conv2DTranspose(filters=?,kernel_size=(3,3),padding="same")
    BatchNormalization(momentum=0.8)
"""
)
bottleneck = Block().define(
"""
    Flatten()
    Dense(units=?,activation="linear")
"""
)
inp = Input(shape = train.X.shape[1:])
x = conv(filters=64) (inp)
x = conv(filters=128) (x)
x = conv(filters=256) (x)
x = bottleneck(units=2) (x)
encoder = Model(inp,x)

inp = Input(shape = encoder.output_shape[1:])
x = Dense(16*16*256) (inp)
x = Reshape((16,16,256)) (x)
x = deconv(filters=256) (x)
x = deconv(filters=128) (x)
x = deconv(filters=64) (x)
x = Conv2D(filters=1,kernel_size=(1,1),padding="same") (x)
decoder = Model(inp,x)

net = Sequential()
net.add(encoder)
net.add(decoder)
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
