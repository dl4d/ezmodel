import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import binarize,keep,split,show_images
from ezmodel.ezblocks import *
from ezmodel.eznetwork import VGG16,VGG19,Xception,MobileNet,InceptionV3,ResNet50,MobileNetV2
import numpy as np

# [EZSET]
parameters = {
    "path": "C:\\Users\\daian\\Desktop\\DATA\\Skin_raw\\skin_128_128.npz",
}
data = ezset(parameters)

# Create a new dataset containing only "non-melanoma" and "mel" images
data_binarized = binarize(
                    data,
                    class0=["nv","bkl","akiec","bcc","df","vasc"],
                    class0_label="non-melanoma",
                    class1="mel",
                    class1_label="melanoma"
                    )

# Save the new dataset in npz format
data_binarized.to_npz("C:\\Users\\daian\\Desktop\\DATA\\Skin_raw\\skin_binary_all_vs_mel_128_128.npz")
