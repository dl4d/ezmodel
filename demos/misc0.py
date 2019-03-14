import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split,keep
import numpy as np

# [EZSET]
parameters = {
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\Skin_raw\\train\\",
    "index"  : "C:\\Users\\daian\\Desktop\\DATA\\Skin_raw\\train.csv",
    "image.path.column" : "image_id",
    "target.column" : "dx",
    "resize"        : (224,224)
}
data = ezset(parameters)

# Create a new dataset containing only "nv" and "mel" images
data_restricted = keep(data,["nv","mel"])

# Undersampling taking minority class number of example as base
a,count = np.unique(data_restricted.y,return_counts=True)
data_restricted.undersampling(min=count.min())

# Save the new dataset in npz format
data_restricted.to_npz("C:\\Users\\daian\\Desktop\\DATA\\Skin_raw\\skin_binary_undersampled_"+str(parameters["resize"][0])+"_"+str(parameters["resize"][0])+".npz")
