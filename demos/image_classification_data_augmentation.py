import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

from ezmodel.ezmodel   import ezmodel
from ezmodel.ezdata    import ezdata
from ezmodel.eztrainer import eztrainer
# [EZ MODEL initialization]
ez_model = ezmodel(type="classification")

# ----------------------------  [EZ Data]  -----------------------------------
# Image Classification from a path directory:
# - One subdirectory by Class
parameters = {
    "name"        : "Bacteria",
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\bacteria\\",
    "resize"      : (32,32)
}

ez_data = ezdata(parameters)

ez_data.gen_test(size=0.2)

ez_data.preprocess(X="minmax",y="categorical")

ez_data.show_images(n=16)

# --------------------------  [EZ Trainer] ------------------------------------
ez_trainer = eztrainer()

ez_trainer.gen_trainval(ez_data,size=0.2)

# Import LeNet5 network from ez_trainer
import keras
ez_trainer.Network(name="LeNet5")

# -- Keras optimizer --
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-4),
    "loss" : "categorical_crossentropy",
    "metrics": ["accuracy"]
}

ez_trainer.compile(optimizer=optimizer)

# Add a data augmentation to the trainer using keras ImageDataGenerator parameters
parameters = {
    "rotation_range" : 15,
    "width_shift_range" : .15,
    "height_shift_range" : .15,
    "horizontal_flip"  : True
}

ez_trainer.keras_augmentation(parameters)


# --------------------------  [EZ Assigment] ----------------------------------
ez_model.assign(ez_data,ez_trainer)


# --------------------------  [EZ Training] -----------------------------------

parameters = {
    "epochs" : 20
}

ez_model.train(parameters)

# # --------------------------  [EZ Evaluation] ---------------------------------
# ez_model.evaluate()
ez_model.learning_graph()

#
# # --------------------------     [EZ Save]    ---------------------------------
# ez_model.save("model0")
