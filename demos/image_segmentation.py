import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))


from ezmodel.ezmodel   import ezmodel
from ezmodel.ezdata    import ezdata
from ezmodel.eztrainer import eztrainer, ezoptimizer



# ----------------------------  [EZ Data]  -----------------------------------
# Image Segmentation from a path directory:
# - One subdirectory by Class

parameters={
    "name"      : "Iris",
    "path"      : "C:\\Users\\daian\\Desktop\\DATA\\Blob\\images\\",
    "path_mask" : "C:\\Users\\daian\\Desktop\\DATA\\Blob\\masks\\",
    "resize"    : (64,64)
}

ez_data = ezdata(parameters)

ez_data.gen_test(size=0.2)

ez_data.preprocess(X="minmax",y="minmax") #on crée les scaler dans ez_data aussi

# --------------------------  [EZ Trainer] ------------------------------------
ez_trainer = eztrainer()

ez_trainer.gen_trainval(ez_data,size=0.2)

import keras

#Make the segmentation network automatically using unet
ez_trainer.Network(name="unet")

# -- Keras optimizer --
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-5),
    "loss"      : ezoptimizer().loss("dice_coefficient"),
    "metrics"   : ezoptimizer().metrics("dice_coefficient")
}

ez_trainer.compile(optimizer=optimizer)


# --------------------------  [EZ Assigment] ----------------------------------
ez_model = ezmodel(type="segmentation")
ez_model.assign(ez_data,ez_trainer)

# --------------------------  [EZ Training] -----------------------------------
parameters = {
    "epochs" : 100
}
ez_model.train(parameters) #Check that we have both ez_data and ez_trainer set

# # --------------------------  [EZ Evaluation] ---------------------------------
ez_model.evaluate()
#
# # --------------------------     [EZ Save]    ---------------------------------
ez_model.save("blob")
