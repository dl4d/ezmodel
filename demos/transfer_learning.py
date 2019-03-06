import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel

from ezmodel.ezutils import split
from ezmodel.eznetwork import Pretrained,MLP,Connect
import keras

# [EZSET]
parameters = {
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\Skin\\skin.npz",
    "X.key"       : "images",
    "y.key"       : "labels",
    "synsets.key" : "synsets"
}
data = ezset(parameters)
# Preprocessing
# NO PREPROCESSING
#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Transform
transformers = train.transform(X="vgg16",y="categorical")
# [EZNETWORK]  ----------------------------------------------------------------
mobilenet = Pretrained(input=train,path="vgg16",frozen=True)
parameters={
    "hidden" : [100,50],
    "activation" : "relu",
    "dropout" : 0.5
}
mlp = MLP(input=train,parameters=parameters,pretrained=mobilenet)
net = Connect(mobilenet,mlp)
# [Keras Optimizer, Loss & Metrics]  ------------------------------------------
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-3),
    "loss" : keras.losses.categorical_crossentropy,
    "metrics" : [keras.metrics.categorical_accuracy]
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
    "epochs" : 5,
    "validation_split": 0.2
}
ez.train(parameters)
# Evaluation ------------------------------------------------------------------
ez.evaluate()
