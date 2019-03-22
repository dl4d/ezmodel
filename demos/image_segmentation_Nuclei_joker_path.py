import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split,bunch
from ezmodel.ezblocks import *
import keras

# [EZSET]
parameters = {
    "path"        : "C:\\Users\\daian\\Documents\\PYTHON\\Nuclei\\data\\train\\***\\images\\",
    "path_mask"   : "C:\\Users\\daian\\Documents\\PYTHON\\Nuclei\\data\\train\\***\\masks\\",
    "resize"      : (128,128),
    "collapse"    : True

}
data = ezset(parameters)

#Save ezset as npz file
data.to_npz("C:\\Users\\daian\\Documents\\PYTHON\\Nuclei\\data\\nuclei.npz")
