import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezutils import split
from ezmodel.ezblocks import *
import keras


# [EZSET]  -------------------------------------------------------------------
parameters = {
    "name"        : "Bacteria",
    "path"        : "C:\\Users\\daian\\Desktop\\DATA\\Bacteria8bit\\",
    "resize"      : (32,32),
    "scaling"     : 1./255,
    "batch_size"  : 32,
    "color_mode"  : "grayscale",
    "class_mode"  : "categorical"
}
data = ezset(parameters,virtual=True)

#Split dataset into Train/Test subset
train,test  = split(data,size=0.2)

# Show 1 images using generator
# x,y = train_gen.next()
# print(x)
# import matplotlib.pyplot as plt
# for i in range(0,1):
#     image = np.squeeze(x[i])
#     plt.imshow(image,cmap="gray")
#     plt.show()
#
# sys.exit()

# # [EZNETWORK]  ----------------------------------------------------------------
from ezmodel.eznetwork import LeNet5
net = LeNet5(input=train)
net.summary()

# [Keras Optimizer, Loss & Metrics]  ------------------------------------------
optimizer = {
    "optimizer" : keras.optimizers.Adam(lr=1e-4),
    "loss" : keras.losses.categorical_crossentropy,
    "metrics" : [keras.metrics.categorical_accuracy]
}

# net.compile(**optimizer)
# [EZMODEL]  ------------------------------------------------------------------
ez = ezmodel(
    train = train,
    test  = test,
    network = net,
    optimizer = optimizer
)

# Training --------------------------------------------------------------------
parameters = {
    "epochs" : 50,
}
ez.train(parameters)

# Evaluation ------------------------------------------------------------------
#ez.evaluate()
# p = ez.network.predict_generator(ez.data_test.generator,steps=ez.data_test.generator.n//ez.data_test.generator.batch_size)
# print(p.argmax(axis=1))
# p=net.predict_generator(test_gen,steps=test_gen.n//test_gen.batch_size)
# print(p)
