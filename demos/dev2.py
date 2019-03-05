import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

from ezmodel.ezmodel   import ezmodel
from ezmodel.ezdata    import ezdata
from ezmodel.eztrainer import eztrainer

# [EZDATA]
parameters = {
    "name"        : "Skin to predict",
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\Skin\\skin_to_predict.npz",
    "resize"      : (128,128)
}

data = ezdata(parameters)
