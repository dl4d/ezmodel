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

# [EZNETWORK with custome EZBLOCKS for making two branches mobilenet network]

inp = Input(shape=train.X.shape[1:])
m0 = PretrainedBlock(path="mobilenet",include_top=False,frozen=False,pooling="avg").get(input_shape=train.X.shape[1:])[0]
m0.name='MobileNet'
m1 = PretrainedBlock(path="mobilenet",include_top=False,frozen=True,pooling="avg").get(input_shape=train.X.shape[1:])[0]
m1.name='FrozenMobileNet'
x = Multiply()([m0(inp), m1(inp)])
out = Dense(2, activation='softmax')(x)
model = Model(inputs=inp, outputs=out)
model.summary()



















###
