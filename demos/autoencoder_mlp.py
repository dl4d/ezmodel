import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split
from ezmodel.ezblocks import *

import keras

# [EZSET]
parameters = {
    "name"        : "Bacteria CSV",
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\bacteria_csv\\bacteria.csv",
    "table.target.column": "Label",
}
data = ezset(parameters)

# Preprocessing : data.y = np.copy(data.X)
data.autoencoder()

#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Transform
transformers = train.transform(X="standard",y="standard")

# [EZNETWORK with custome EZBLOCKS]

# An example of Autoencoder
dense1    = DenseBlock(units=1500,activation="relu",dropout=0.5)
dense2    = DenseBlock(500,activation="sigmoid")
bottleneck= DenseBlock(100)
net       = ConnectAE(input=train,transformers=transformers,blocks=[dense1,dense2,bottleneck])
net.summary()

# [Keras Optimizer, Loss & Metrics]  ------------------------------------------
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-4),
    "loss" : keras.losses.mean_squared_error,
    "metrics" : [keras.metrics.mean_squared_error]
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
    "epochs" : 100,
    # "validation_split": 0.2
}
ez.train(parameters)
# Evaluation ------------------------------------------------------------------
ez.evaluate()

# Prediction
p = ez.predict()
print(p[0][0:10])

input0 = copy.deepcopy(test)
input0.preprocess(X=transformers[0],y=transformers[1])
print(input0.y[0][0:10])
