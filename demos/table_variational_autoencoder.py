import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split,show_images
from ezmodel.ezblocks import *
from ezmodel.eznetwork import VAE
from ezmodel import ezlosses

import keras

# [EZSET] ---------------------------------------------------------------------
parameters = {
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\bacteria_csv\\bacteria.csv",
    "table.target.column" : "Label"
}
data = ezset(parameters)
data.autoencoder()

#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Create Transformers on training set (further be used for test set when evaluated)
transformers = train.transform(X="standard",y="standard")

# [EZNETWORK]  ----------------------------------------------------------------
original_dim = train.X.shape[1]
intermediate_dim = 128
latent_dim = 100

parameters={
    "intermediate_dim" : 128,
    "latent_dim"       : 100,
    "activation"       : "relu",
    "dropout"          : 0.5,
    "bn"               : 0.8
}

vae , z_mean, z_log_var = VAE(input=train,transformers=transformers,parameters=parameters)

# [Keras Optimizer, Loss & Metrics]  ------------------------------------------
optimizer ={
    "optimizer" : keras.optimizers.Adam(lr=1e-3),
    "loss"      : ezlosses.vae_loss(z_mean,z_log_var,original_dim),
    "metrics"   : [ezlosses.reconstruction_loss]
}

# [EZMODEL]  ------------------------------------------------------------------
ez = ezmodel(
    train = train,
    test  = test,
    network = vae,
    optimizer = optimizer,
    transformers = transformers
)

# Training --------------------------------------------------------------------
parameters = {
    "epochs" : 200
}
ez.train(parameters)

ez.learning_graph()

p = ez.predict()
print(p[0][0:10])
print(test.X[0][0:10])
