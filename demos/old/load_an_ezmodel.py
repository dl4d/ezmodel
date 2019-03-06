import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

from ezmodel.ezmodel   import ezmodel
from ezmodel.ezdata    import ezdata
from ezmodel.eztrainer import eztrainer

# [EZ MODEL loading]
ez_model = ezmodel(load = "model0")

print("Model parameters = ",ez_model.model_parameters)

# [EZ evaluation]
ez_model.evaluate()
