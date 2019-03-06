import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel

from ezmodel.ezutils import split
from ezmodel.eznetwork import LeNet5,MLP
import keras

# [EZSET]  -------------------------------------------------------------------
parameters = {
    "name"        : "Bacteria CSV",
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\bacteria_csv\\bacteria.csv",
    "table.target.column": "Label",
}
data = ezset(parameters)
# Preprocessing
# data.dropfeature(columns=["Id"])
#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Transform
transformerX,transformerY = train.transform(X="standard",y="categorical")


# [EZNETWORK]  ----------------------------------------------------------------
parameters = {
    "hidden" : [100,50,30],
    "activation" : "relu",
    "dropout" : 0.5
}
net = MLP(input=train,parameters=parameters)
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
    transformers = (transformerX,transformerY)
)
# Training --------------------------------------------------------------------
parameters = {
    "epochs" : 50,
    "validation_split": 0.2
}
ez.train(parameters)
# Evaluation ------------------------------------------------------------------
ez.evaluate()
