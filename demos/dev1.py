import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel

from ezmodel.ezutils import split,show_images
from ezmodel.ezblocks import *
from ezmodel.eznetwork import LeNet5
from ezmodel import ezlosses
import keras

# [EZSET]
parameters = {
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\Skin\\skin.npz",
    "X.key"       : "images",
    "y.key"       : "labels",
    "synsets.key" : "synsets"
}
data = ezset(parameters)

#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Transform
transformers = train.transform(X="standard",y="categorical")

# [EZNETWORK with custome EZBLOCKS]

#LeNet5
# conv1 = ConvBlock(filters=6,kernel_size=(5,5),activation="relu",pooling=(2,2))
# conv2 = conv1.copy(filters=16)
# dense1 = DenseBlock(units=120)
# dense2 = dense1.copy(units=84)
# net = ConnectBlock(input=train,transformers=transformers,blocks=[conv1,conv2,dense1,dense2])
# net.summary()

#One Inception Block
incept = InceptionBlock(filters=64,activation="relu")
dense = DenseBlock(units=10) (incept)
#net = ConnectBlock(input=train,transformers=transformers,blocks=[incept,dense])
#net.summary()
















###
