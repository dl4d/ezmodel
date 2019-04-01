import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split,show_images
from ezmodel.ezblocks import *

import keras

# [EZSET]  -------------------------------------------------------------------
parameters={
    # "name"      : "Mitochondria",
    "path"      : "C:\\Users\\daian\\Desktop\\DATA\\Mito\\images\\",
    "path_mask" : "C:\\Users\\daian\\Desktop\\DATA\\Mito\\masks\\",
    "resize"    : (128,128)
}
data = ezset(parameters)
data.autoencoder()
#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Transform
transformers = train.transform(X="standard",y="minmax")

# [EZNETWORK with custome EZBLOCKS]
from ezmodel.eznetwork import UNET2
parameters={
    "n": 16,
    "depth": 6
}
unet = UNET2(input=train,transformers=transformers,parameters=parameters)
unet.summary()


# [Keras Optimizer, Loss & Metrics]  ------------------------------------------
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-3),
    "loss"      : keras.losses.mean_squared_error,
    "metrics"   : [keras.metrics.mean_squared_error]
}
# [EZMODEL]  ------------------------------------------------------------------
ez = ezmodel(
    train = train,
    test  = test,
    network = unet,
    optimizer = optimizer,
    transformers = transformers
)
# Training --------------------------------------------------------------------
parameters = {
    "epochs" : 40,
    # "validation_split": 0.2
}
ez.train(parameters)


p = ez.predict()
r = show_images(p,4)

input0 = copy.deepcopy(test)
input0.preprocess(X=transformers[0],y=transformers[1])
show_images(input0.y,4,samples=r)
