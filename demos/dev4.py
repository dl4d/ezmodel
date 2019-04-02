import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split,show_images
from ezmodel.ezblocks4 import *
from ezmodel.eznetwork import reparam_trick_vae
from keras.models import Model
from ezmodel import ezlosses
from ezmodel.ezgan import ezgan

import keras
import numpy as np

# [EZSET]  -------------------------------------------------------------------
parameters = {
    "name"        : "Bacteria",
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\bacteria\\",
    "resize"      : (32,32)
}
data = ezset(parameters)
#Transform
transformers = data.transform(X="minmax",y=None)

# [EZGAN] ---------------------------------------------------------------------
gan = ezgan(
        data=data,
        transformers=transformers,
        parameters=parameters
)
# Generator -------------------------------------------------------------------
parameters = {
    "noise_dim" : 100,
    "n" : 1024,
    "depth"     : 4
}
gan.basic_generator(data=data,parameters=parameters)

# Discriminator ---------------------------------------------------------------
parameters = {
    "n": 512,
    "depth": 4,
}
gan.basic_discriminator(data=data,parameters=parameters)

# Gan -------------------------------------------------------------------------
optimizer_gan = {
    "optimizer" : keras.optimizers.RMSprop(lr=0.005),
    "loss"      : ezlosses.wasserstein_loss,
    "metrics"   : ["accuracy"]
}
gan.generate(optimizer=optimizer_gan)

# Training
parameters = {
    "epochs": 100,
    "batch_size": 32,
    "n_discriminator": 15,
    "discriminator_clip_value" : 0.01,
    # "logdir" : "C:\\Users\\daian\\Desktop\\LOGDIR\\GAN-BACTERIA\\"
    "logdir" : None,
    "show" : True
}
gan.train(parameters)
