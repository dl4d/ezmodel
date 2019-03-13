import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split
from ezmodel.ezblocks import *
from ezmodel.eznetwork import VGG16,VGG19,Xception,MobileNet,InceptionV3,ResNet50,MobileNetV2


# [EZSET]
parameters = {
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\Skin_raw\\train\\",
    "index"  : "C:\\Users\\daian\\Desktop\\DATA\\Skin_raw\\train.csv",
    "image.path.column" : "image_id",
    "target.column" : "dx",
    "resize"        : (128,128)
}
data = ezset(parameters)
data.to_npz("C:\\Users\\daian\\Desktop\\DATA\\Skin_raw\\skin_"+str(parameters["resize"][0])+"_"+str(parameters["resize"][0])+".npz")
