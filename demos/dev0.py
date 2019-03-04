import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

from ezmodel.ezmodel   import ezmodel
from ezmodel.ezdata    import ezdata
from ezmodel.eztrainer import eztrainer

# ----------------------------  [EZ Data]  -----------------------------------
# Load from a npz file containing data
# NPZ file should contains 3 keys:
# X key containing data
# y key containing label
# synsets key in case of classification model with synset generated
parameters = {
    "name"        : "Bacteria",
    "path"        : ".\\bacteria.npz",
}

ez_data = ezdata(parameters)
