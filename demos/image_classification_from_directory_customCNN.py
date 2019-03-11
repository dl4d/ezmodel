import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel

from ezmodel.ezutils import split,show_images
from ezmodel.eznetwork import LeNet5,MLP
import keras

# [EZSET]  -------------------------------------------------------------------
parameters = {
    "name"        : "Bacteria",
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\bacteria\\",
    "resize"      : (32,32)
}
data = ezset(parameters)
#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Create Transformers on training set (further be used for test set when evaluated)
transformers = train.transform(X="standard",y="categorical")
# [EZNETWORK]  ----------------------------------------------------------------
#net = LeNet5(input=train,transformers=transformers)
from ezmodel.eznetwork import ConvolutionBlock

conv1 = ConvBlock(filters=16,
                             kernel=(5,5),
                             stride=(1,1),
                             padding="valid",
                             activation="relu",
                             pooling=(2,2),
                             bn = None
                             ),
conv2 = conv1.copy(filter=6)

dense = DenseBlock(n=100,activation="relu",dropout=0.5,bn=None)

dense2 = copyDenseBlock(dense,n=20)

net = ConnectBlock([conv1,conv2,dense,dense2])




# # [Keras Optimizer, Loss & Metrics]  ------------------------------------------
# optimizer = {
#     "optimizer" : keras.optimizers.Adam(lr=1e-4),
#     "loss" : keras.losses.categorical_crossentropy,
#     "metrics" : [keras.metrics.categorical_accuracy]
# }
# # [EZMODEL]  ------------------------------------------------------------------
# ez = ezmodel(
#     train = train,
#     test  = test,
#     network = net,
#     optimizer = optimizer,
#     transformers = transformers
# )
# # Training --------------------------------------------------------------------
# parameters = {
#     "epochs" : 50,
#     "validation_split": 0.2
# }
# ez.train(parameters)
# # Evaluation ------------------------------------------------------------------
# ez.evaluate()
