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

# optimizer:
#
# Note : The two losses are combined (sum) together
#
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


# Training --------------------------------------------------------------------

model.fit(train0.X,
          {'classification': train0.y, 'cae': train0.X},
          epochs=5,
          validation_data= (test0.X, {'classification': test0.y, 'cae': test0.X}),
          verbose=1)
