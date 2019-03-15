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
#
# datagen = keras.preprocessing.image.ImageDataGenerator(
#         rescale=1./255,
#         validation_split=0.1
# )
#
# train_gen = datagen.flow_from_directory(
#                         data.params["path"],
#                         target_size=data.params["resize"],
#                         batch_size=data.params["batch_size"],
#                         color_mode=data.params["color_mode"],
#                         class_mode=data.params["class_mode"],
#                         # shuffle=True,
#                         subset="training")
#
# test_gen = datagen.flow_from_directory(
#                         data.params["path"],
#                         target_size=data.params["resize"],
#                         batch_size=data.params["batch_size"],
#                         color_mode=data.params["color_mode"],
#                         class_mode=data.params["class_mode"],
#                         # shuffle=True,
#                         subset="validation")

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
# net = Sequential()
# net.add(Conv2D(6,(5,5),input_shape=(32,32,1)))
# net.add(Activation("relu"))
# net.add(MaxPooling2D(pool_size=(2,2)))
# net.add(Conv2D(16,(5,5)))
# net.add(Activation("relu"))
# net.add(MaxPooling2D(pool_size=(2,2)))
# net.add(Flatten())
# net.add(Dense(120))
# net.add(Dense(80))
# net.add(Dense(4,activation="softmax"))
# net.summary()


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
# history = net.fit_generator(
#                 train_gen,
#                 steps_per_epoch=train_gen.samples//train_gen.batch_size,
#                 validation_data = test_gen,
#                 validation_steps = test_gen.samples//test_gen.batch_size,
#                 epochs=50,
#                 verbose = 1,
#                 callbacks=None
#                 )


# Evaluation ------------------------------------------------------------------
#ez.evaluate()
# p = ez.network.predict_generator(ez.data_test.generator,steps=ez.data_test.generator.n//ez.data_test.generator.batch_size)
# print(p.argmax(axis=1))
# p=net.predict_generator(test_gen,steps=test_gen.n//test_gen.batch_size)
# print(p)
