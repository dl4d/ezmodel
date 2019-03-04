import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

from ezmodel.ezmodel   import ezmodel
from ezmodel.ezdata    import ezdata
from ezmodel.eztrainer import eztrainer

# ----------------------------  [EZ Data]  -----------------------------------
# Image Classification from a path directory:
# - One subdirectory by Class
parameters = {
    "name"        : "Bacteria",
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\bacteria\\",
    "resize"      : (32,32)
}

ez_data = ezdata(parameters)

ez_data.to_npz("bacteria")
