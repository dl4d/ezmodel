import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel

from ezmodel.ezutils import split
from ezmodel.eznetwork import LeNet5
import keras

# [EZSET]
parameters = {
    "path": "C:\\Users\\daian\\Desktop\\DATA\\Skin_raw\\skin_binary_undersampled_128_128.npz",
}
data = ezset(parameters)

# Preprocessing
# NO PREPROCESSING
#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Transform
transformers = train.transform(X="standard",y="categorical")
# [EZNETWORK]  ----------------------------------------------------------------
net = LeNet5(input=train,transformers=transformers)
# [Keras Optimizer, Loss & Metrics]  ------------------------------------------
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-4),
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
    "epochs" : 5
}
ez.train(parameters)
# Evaluation ------------------------------------------------------------------
#ez.evaluate()
