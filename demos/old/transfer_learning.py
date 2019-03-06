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

ez_data.gen_test(0.2)

#ez_data.preprocess(X="mobilenet",y="categorical")
ez_data.preprocess(X="vgg19",y="categorical")


# --------------------------  [EZ Trainer] ------------------------------------
from keras.layers import Dense,Dropout

ez_trainer = eztrainer()

ez_trainer.gen_trainval(ez_data,size=0.2)

#  -- Keras network --

#mobilenet = ez_trainer.Transfer(name="mobilenet",frozen=True)
mobilenet = ez_trainer.EZTransfer(name="vgg19",frozen=True)

inputs = ez_trainer.Input(transfer_model = mobilenet)
x = Dense(1024)(inputs)
x = Dropout(0.5) (x)
x = Dense(1024)(x)
outputs = ez_trainer.ClassificationOutput(x)

ez_trainer.gen_network(inputs=inputs,outputs=outputs,transfer_model=mobilenet)

ez_trainer.network.summary()


# # -- Keras optimizer --
import keras

optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-4),
    "loss" : "categorical_crossentropy",
    "metrics": ["accuracy"]
}

ez_trainer.compile(optimizer=optimizer)


# --------------------------  [EZ Assigment] ----------------------------------
ez_model = ezmodel()
ez_model.assign(ez_data,ez_trainer)
