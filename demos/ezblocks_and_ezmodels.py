import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split
from ezmodel.ezblocks import *
from ezmodel.eznetwork import VGG16,VGG19,Xception,MobileNet,InceptionV3,ResNet50

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
conv1  = ConvBlock(filters=6,kernel_size=(5,5),activation="relu",pooling=(2,2))
conv2  = ConvBlock(filters=16,kernel_size=(5,5),activation="relu",pooling=(2,2))
dense1 = DenseBlock(units=120)
dense2 = DenseBlock(units=80)
net    = ConnectBlock(input=train,transformers=transformers,blocks=[conv1,conv2,dense1,dense2])
net.summary()

#Inceptions Blocks
incept  = InceptionBlock(filters=64,activation="relu")
incept2 = NaiveInceptionBlock(filters=6,activation="relu")
incept3 = InceptionBlockDimReduce(filters=36,activation="relu")
dense   = DenseBlock(units=10)
net     = ConnectBlock(input=train,transformers=transformers,blocks=[incept,incept2,incept3,dense])
net.summary()


# [EZMODEL inherited from Keras and converted to ezmodel]

model = VGG16(input=train,transformers=transformers,parameters=None)
model.summary()

model = VGG19(input=train,transformers=transformers,parameters=None)
model.summary()

model = Xception(input=train,transformers=transformers,parameters=None)
model.summary()

model = MobileNet(input=train,transformers=transformers,parameters=None)
model.summary()

model = InceptionV3(input=train,transformers=transformers,parameters=None)
model.summary()

model = ResNet50(input=train,transformers=transformers,parameters=None)
model.summary()














###
