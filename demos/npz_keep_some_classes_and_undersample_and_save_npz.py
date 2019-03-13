import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import keep,split,show_images
from ezmodel.ezblocks import *
from ezmodel.eznetwork import VGG16,VGG19,Xception,MobileNet,InceptionV3,ResNet50,MobileNetV2

# [EZSET]
parameters = {
    "path": "C:\\Users\\daian\\Desktop\\DATA\\Skin_raw\\skin_128_128.npz",
}
data = ezset(parameters)

# Create a new dataset containing only "nv" and "mel" images
data_restricted = keep(data,["nv","mel"])

a,count = np.unique(data_restricted.y,return_counts=True)

data_restricted.undersampling(min=count.min())

# Save the new dataset in npz format
data_restricted.to_npz("C:\\Users\\daian\\Desktop\\DATA\\Skin_raw\\skin_binary_undersampled_128_128.npz")
