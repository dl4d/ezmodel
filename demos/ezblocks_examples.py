import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split
from ezmodel.ezblocks import *

# [EZSET]
parameters = {
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\Skin_raw\\skin_binary_undersampled_128_128.npz",
}
data = ezset(parameters)

#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Transform
transformers = train.transform(X="standard",y="categorical")

# [EZNETWORK with custome EZBLOCKS]

#LeNet5 written with mix ezblocks + Keras Layer
conv1  = ConvBlock(filters=6,kernel_size=(5,5),activation="relu",pooling=(2,2))
conv2  = ConvBlock(filters=16,kernel_size=(5,5),activation="relu",pooling=(2,2))
pooling = GlobalAveragePooling2D()
dense1 = DenseBlock(units=120)
dense2 = Dense(80)
net    = Connect(input=train,transformers=transformers,blocks=[conv1,conv2,pooling,dense1,dense2])
net.summary()

#Transfer learning from VGG16 written with ezblocks
pretrained = PretrainedBlock(path="vgg16",include_top=False,frozen=False,pooling="avg")
dense      = DenseBlock(units=200,activation="relu",dropout=0.5)
net = Connect(input=train,transformers=transformers,blocks=[pretrained,dense])
net.summary()















###
