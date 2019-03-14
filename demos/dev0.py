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


from ezmodel.ezblocks import *
pretrained = PretrainedBlock(path="vgg16",include_top=False,frozen=True,pooling="avg")
dense      = DenseBlock(units=4096)
cnn        = Connect(input=train,transformers=transformers,blocks=[pretrained,dense])
