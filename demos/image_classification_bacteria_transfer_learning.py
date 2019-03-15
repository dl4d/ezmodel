import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel

from ezmodel.ezutils import split,show_images
from ezmodel.eznetwork import LeNet5
from ezmodel.ezblocks import *
import keras
import numpy as np


# [EZSET]  -------------------------------------------------------------------
parameters = {
    "name"        : "Bacteria",
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\bacteria\\",
    "resize"      : (32,32)
}
data = ezset(parameters)

#Preprocessing
data.falseRGB()

#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)

#Create Transformers on training set (further be used for test set when evaluated)
transformers = train.transform(X="standard",y="categorical")

# [EZNETWORK]  ----------------------------------------------------------------
#Transfer
pretrained = PretrainedBlock(path="vgg16",include_top=False,frozen=False,pooling="avg")
dense      = DenseBlock(units=1000,activation="relu",dropout=0.5)
net = Connect(input=train,transformers=transformers,blocks=[pretrained,dense])
net.summary()
# [Keras Optimizer, Loss & Metrics]  ------------------------------------------
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-5),
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
    "epochs" : 50,
    "callbacks" : [keras.callbacks.EarlyStopping(monitor="val_loss",patience=5)]
}
ez.train(parameters)
# Evaluation ------------------------------------------------------------------
ez.evaluate()
ez.learning_graph()
ez.ROC()
