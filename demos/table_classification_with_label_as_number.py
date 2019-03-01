import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

from ezmodel.ezmodel   import ezmodel
from ezmodel.ezdata    import ezdata
from ezmodel.eztrainer import eztrainer
# [EZ MODEL initialization]
ez_model = ezmodel(type="classification")
print(ez_model.type)

# ----------------------------  [EZ Data]  -----------------------------------
# Table Classification from a csv file:
# - One subdirectory by Class
parameters = {
    "name"                      : "Bacteria CSV",
    "type"                      : "classification",
    "format"                    : "table",
    "from"                      : "file",
    "path"                      : "C:\\Users\\daian\\Desktop\\DATA\\bacteria_csv\\bacteria.csv",
    "table_delimiter"           : "\t",
    "table_target_column"       : ["Label"],
    "table_target_column_type"  : "number"
    "table_drop_column"         : ["Id"]
}

ez_data = ezdata(parameters)

ez_data.gen_test(size=0.2)

ez_data.preprocess(X="standard",y="categorical") #on crée les scaler dans ez_data aussi

# --------------------------  [EZ Trainer] ------------------------------------
ez_trainer = eztrainer()

ez_trainer.gen_trainval(ez_data,size=0.2)

import keras
from keras.layers import Dense

#  -- Keras network --
inputs = ez_trainer.Input()
x = Dense(100) (inputs)
x = Dense(20) (x)
outputs = ez_trainer.ClassificationOutput(x)

ez_trainer.gen_network(inputs,outputs)

# -- Keras optimizer --
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-3),
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
ez_model.train(parameters) #Check that we have both ez_data and ez_trainer set

# --------------------------  [EZ Evaluation] ---------------------------------
ez_model.evaluate()

# --------------------------     [EZ Save]    ---------------------------------
ez_model.save("model0")
