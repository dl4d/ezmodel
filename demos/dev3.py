import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split,show_images
from ezmodel.ezblocks4 import *
from ezmodel.eznetwork import reparam_trick_vae
from keras.models import Model
from ezmodel import ezlosses

import keras

# [EZSET]  -------------------------------------------------------------------
parameters={
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
#Variational Unet

latent_dim = 100
depth = 6
n = 16

conv = Block().define(
    """
    Conv2D(filters=?,kernel_size=(3,3),activation="relu",padding="same")
    Conv2D(filters=?,kernel_size=(3,3),activation="relu",padding="same")
    """
)

inputs = Input(shape=train.X.shape[1:])

x = inputs

#Encoder
convs=[]
for i in range(depth):
    x = conv(filters=n*(2**i)) (x)
    convs.append(x)
    x = MaxPooling2D(pool_size=(2,2)) (x)

#bottleneck
x = conv(filters=n*(2**(i+1))) (x)
last_shape = K.int_shape(x)

x = Flatten() (x)

z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Lambda(reparam_trick_vae, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

x = Dense(np.prod(last_shape[1:]),activation="relu", name="first_decoder") (z)
x = Reshape((last_shape[1:])) (x)

# #Decoder
for i in reversed(range(depth)):
    x = Conv2DTranspose(n*(2**i), (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([x, convs[i]], axis=3)
    x = conv(filters=n*(2**i))(x)

#Output
outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

var_unet = Model(inputs, outputs)
var_unet.summary()

#[Keras Optimizer, Loss & Metrics]  ------------------------------------------
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-3),
    "loss"      : ezlosses.vae_loss(z_mean,z_log_var,train.X.shape[1:]),
    # "metrics"   : [ezlosses.reconstruction_loss]
}

# # [EZMODEL]  ------------------------------------------------------------------
ez = ezmodel(
    train = train,
    test  = test,
    network = var_unet,
    optimizer = optimizer,
    transformers = transformers
)
# # # Training --------------------------------------------------------------------
parameters = {
    "epochs" : 40,
    # "validation_split": 0.2
}
ez.train(parameters)



p = ez.predict()
r = show_images(p,4)

input0 = copy.deepcopy(test)
input0.preprocess(X=transformers[0],y=transformers[1])
show_images(input0.y,4,samples=r)
