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
data_to_predict = ezdata(parameters)

# [EZNETWORK]
ez_trainer = eztrainer(network="imagenet.mobilenet")

# [EZMODEL]

ez_model = ezmodel()
ez_model.assign(data_to_predict,ez_trainer)

ez_model.predict(data_to_predict)
