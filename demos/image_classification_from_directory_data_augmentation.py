import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel

from ezmodel.ezutils import split,show_images
from ezmodel.eznetwork import LeNet5
import keras

# [EZSET]  -------------------------------------------------------------------
parameters = {
    "name"        : "Bacteria",
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\bacteria\\",
    "resize"      : (32,32)
}
data = ezset(parameters)
#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)
#Create Transformers on training set (further be used for test set when evaluated)
transformers = train.transform(X="standard",y="categorical")
# [EZNETWORK]  ----------------------------------------------------------------
net = LeNet5(input=train,transformers=transformers)
# [Keras Optimizer, Loss & Metrics]  ------------------------------------------
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-4),
    "loss" : keras.losses.categorical_crossentropy,
    "metrics" : [keras.metrics.categorical_accuracy]
}
# [EZMODEL]  ------------------------------------------------------------------
augmentation_parameters={
    "rotation_range" : 15,
    "width_shift_range" : .15,
    "height_shift_range" : .15,
    "horizontal_flip"  : True
}

ez = ezmodel(
    train = train,
    test  = test,
    network = net,
    optimizer = optimizer,
    transformers = transformers,
    augmentation = augmentation_parameters
)

# Training --------------------------------------------------------------------
parameters = {
    "epochs" : 200,
    # "validation_split": 0.2
}
ez.train(parameters)
# Evaluation ------------------------------------------------------------------
ez.evaluate()

# Evaluation already defined augmentation for Test Time Augmentation (TTA)
ez.evaluate(tta=10)

# Evaluation with newly Test Time Augmentation (TTA)
augmentation_parameters={
    "rotation_range" : 5,
    "width_shift_range" : .15,
    "height_shift_range" : .15,
    "horizontal_flip"  : True
}
ez.evaluate(tta=10,augmentation=augmentation_parameters)

# ez.learning_graph()
# ez.confusion_matrix()

# Predictions
# p     = ez.predict()

# Prediction with Test Time Augmentation (TTA)
# p_TTA = ez.predict(tta=10)

#
