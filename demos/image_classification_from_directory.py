import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

from ezmodel.ezmodel   import ezmodel
from ezmodel.ezdata    import ezdata
from ezmodel.eztrainer import eztrainer
# [EZ MODEL initialization]
ez_model = ezmodel(type="classification")
print(ez_model.type)

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

ez_data.preprocess(X="minmax",y="categorical") #on crée les scaler dans ez_data aussi

# --------------------------  [EZ Trainer] ------------------------------------
ez_trainer = eztrainer()

ez_trainer.gen_trainval(ez_data,size=0.2)

import keras
from keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dense



#  -- Keras network --
inputs = ez_trainer.Input()
x = Conv2D(6, kernel_size = (5, 5), strides=(1,1), padding="valid",input_shape=(32, 32, 1)) (inputs)
x = Activation("relu") (x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (x)
x = Conv2D(16, kernel_size = (5, 5), strides=(1,1), padding="valid") (x)
x = Activation("relu") (x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (x)
x = Flatten() (x)
x = Dense(120) (x)
x = Dense(84) (x)
outputs = ez_trainer.ClassificationOutput(x)

ez_trainer.gen_network(inputs,outputs)

# -- Keras optimizer --
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-4),
    "loss" : "categorical_crossentropy",
    "metrics": ["accuracy"]
}

ez_trainer.compile(optimizer=optimizer)

# --------------------------  [EZ Assigment] ----------------------------------
ez_model.assign(ez_data,ez_trainer)


# --------------------------  [EZ Training] -----------------------------------
parameters = {
    "epochs" : 20
}

ez_model.train(parameters)

# --------------------------  [EZ Evaluation] ---------------------------------
ez_model.evaluate()

# --------------------------     [EZ Save]    ---------------------------------
ez_model.save("model0")
