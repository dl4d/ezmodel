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
    "path"        : ".\\bacteria.npz",
}

ez_data = ezdata(parameters)
