import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel

from ezmodel.ezutils import split,show_images
from ezmodel.eznetwork import MLP
import keras

# [EZSET]  -------------------------------------------------------------------
parameters = {
    "name"        : "Bacteria Table",
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\bacteria_csv\\bacteria.csv",
    "table.target.column" : "Label"
}
data = ezset(parameters)

#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Create Transformers on training set (further be used for test set when evaluated)
transformers = train.transform(X="minmax",y="categorical")

# # [EZNETWORK]  ----------------------------------------------------------------
parameters = {
    "hidden" : [100],
    "activation" : "sigmoid"
}
net = MLP(input=train,transformers=transformers,parameters=parameters)

net.summary()

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
#
# Training --------------------------------------------------------------------
parameters = {
    "epochs" : 50
}
ez.train(parameters)
# # Evaluation ------------------------------------------------------------------
ez.evaluate()
ez.learning_graph()
