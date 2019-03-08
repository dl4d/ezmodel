import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel

from ezmodel.ezutils import split,show_images
from ezmodel.eznetwork import UNET
from ezmodel import ezlosses
import keras

# [EZSET]  -------------------------------------------------------------------
parameters={
    "name"      : "Blob",
    "path"      : "C:\\Users\\daian\\Desktop\\DATA\\Blob\\images\\",
    "path_mask" : "C:\\Users\\daian\\Desktop\\DATA\\Blob\\masks\\",
    "resize"    : (64,64)
}
data = ezset(parameters)

import matplotlib.pyplot as plt
import numpy as np
plt.imshow(np.squeeze(data.X[10]))
plt.show()
plt.imshow(np.squeeze(data.y[10]))
plt.show()

# #Split dataset into Train/Test subset
# train,test  = split(data,size=0.2)
# #Transform
# transformers = train.transform(X="minmax",y="minmax")
# # [EZNETWORK]  ----------------------------------------------------------------
# parameters = {
#     "n" : 64
# }
# net = UNET(input=train,parameters=parameters)
# # [Keras Optimizer, Loss & Metrics]  ------------------------------------------
# optimizer = {
#     "optimizer" : keras.optimizers.Adam(lr=1e-4),
#     "loss"      : ezlosses.dice_loss,
#     "metrics"   : [ezlosses.dice_metrics,keras.metrics.mean_squared_error]
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
# }
# ez.train(parameters)
# # Evaluation ------------------------------------------------------------------
# ez.evaluate()
# # save
# ez.save("blob")
