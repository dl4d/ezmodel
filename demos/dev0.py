import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split,bunch
from ezmodel.ezblocks import *
import keras

# [EZSET]
parameters = {
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\Skin_raw\\skin_binary_undersampled_128_128.npz",
}
data = ezset(parameters)

#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Transform
transformers = train.transform(X="standard",y="categorical")

# [EZNETWORK]  ----------------------------------------------------------------
from ezmodel.eznetwork import LeNet5,MobileNetV2,MobileNet
from ezmodel.ezlosses import f1_loss,f1_metrics
net = LeNet5(input=train,transformers=transformers)
# [Keras Optimizer, Loss & Metrics]  ------------------------------------------
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-4),
    "loss" : f1_loss,
    "metrics" : [f1_metrics,keras.metrics.categorical_accuracy]
}

network_bunch = NetworkBunch(net=net,opt=optimizer)

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
    "callbacks": [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=5, min_lr=1e-10, verbose=1)]
}
ez.train(parameters)
# Evaluation ------------------------------------------------------------------
ez.ROC()
#ez.learning_graph()
