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
    "name"        : "Skin",
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\Skin\\skin.npz",
    "X.key"       : "images",
    "y.key"       : "labels"
}

ez_data = ezdata(parameters)

ez_data.preprocess(X="mobilenet",y="categorical")


from keras.applications.mobilenet import MobileNet
mobilenet = MobileNet(include_top=False, weights='imagenet', input_shape=ez_data.X.shape[1:])


# --------------------------  [EZ Trainer] ------------------------------------
from keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dense,GlobalAveragePooling2D,Dropout,Input
from keras.models import Sequential, Model

ez_trainer = eztrainer()

ez_trainer.gen_trainval(ez_data,size=0.2)

#  -- Keras network --

inputs,transfer_model = ez_trainer.Input(transfer_model = mobilenet)
x = Dense(1024)(inputs)
x = Dropout(0.5) (x)
x = Dense(1024)(x)
outputs = ez_trainer.ClassificationOutput(x)
ez_trainer.gen_network(inputs=inputs,outputs=outputs,transfer_model=transfer_model)

ez_trainer.network.summary()


# # -- Keras optimizer --
import keras

optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-4),
    "loss" : "categorical_crossentropy",
    "metrics": ["accuracy"]
}

ez_trainer.compile(optimizer=optimizer)
