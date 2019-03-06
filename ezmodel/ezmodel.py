import pickle
from zipfile import ZipFile
import os
import time
from keras.models import load_model
import sys
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
import copy
from keras.preprocessing.image import ImageDataGenerator



class ezmodel:

    def __init__(self,train=None,test=None,network=None,optimizer=None,transformers=None,augmentation=None):

        if train is None:
            raise Exception("ezmodel.init() : Please provide a train dataset !")

        if test is None:
            raise Exception("ezmodel.init() : Please provide a test dataset !")

        if network is None:
            raise Exception("ezmodel.init(): Please provide a Keras network !")

        if optimizer is None:
            raise Exception("ezmodel.init(): Please provide an optimizer !")


        self.data_train   = train
        self.data_test    = test
        self.network = network
        self.optimizer = optimizer
        self.transformerX = transformers[0]
        self.transformerY = transformers[1]
        self.model_parameters = None
        self.history = None
        self.augmentation = None

        self.network.compile(**optimizer)

        if augmentation is not None:
            self.keras_augmentation(augmentation)


    def train(self,parameters=None):
        #default parameters
        epochs = 10
        callbacks = None
        verbose = 1
        batch_size = 32
        validation_data = None

        if parameters is not None:
            if "epochs" in parameters:
                epochs = parameters["epochs"]
            if "callbacks" in parameters:
                callbacks = parameters["callbacks"]
            if "verbose" in parameters:
                verbose = parameters["verbose"]
            if "batch_size" in parameters:
                batch_size = parameters["batch_size"]
            if "validation_split" in parameters:
                X_train,X_valid,y_train,y_valid = train_test_split(self.data_train.X,self.data_train.y,test_size=parameters["validation_split"],random_state=None)
                validation_data = (X_valid,y_valid)

            self.model_parameters = parameters

        if self.augmentation is None:
            history = self.network.fit(
                            self.data_train.X,
                            self.data_train.y,
                            validation_data=validation_data,
                            epochs=epochs,
                            batch_size = batch_size,
                            callbacks=callbacks,
                            verbose = verbose
                            )
        else:
            print("[X] Training with Data augmentation on Training Set.")
            history = self.network.fit_generator(
                            self.augmentation.flow(self.data_train.X,self.data_train.y,batch_size = batch_size),
                            validation_data = validation_data,
                            steps_per_epoch = self.data_train.X.shape[0]//batch_size,
                            epochs=epochs,
                            verbose = verbose,
                            callbacks=callbacks
                            )

        #Save history
        if self.history is None:
            self.history = history.history
        else:
            for key in history.history:
                self.history[key] += history.history[key]


    def evaluate(self):

        print ("[X] Evaluation on Test set: ")
        test = copy.deepcopy(self.data_test)
        test.preprocess(X=self.transformerX,y=self.transformerY)
        print("--- Use transformers to preprocess Test set : Done")
        p = self.network.evaluate(test.X,test.y,verbose=0)
        if "metrics" in self.optimizer:
            print ("--- Loss    : ", p[0])
            print ("--- Metrics : ", p[1])
        else:
            print ("--- Loss    : ", p)
        print("\n")

        return p


    def keras_augmentation(self,parameters):
        image_gen = ImageDataGenerator(**parameters)
        if self.data_train.X is None:
            raise Exception("[Fail] ezmodel.augmentation(): No Training set has been added to this ezmodel object")
        image_gen.fit(self.data_train.X, augment=True)
        self.augmentation = image_gen
        print("[X] Keras ImageDataGenerator has been added to ezmodel")
        print("\n")

    # def confusion_matrix(self):
    #     from IPython.display import display
    #     print("[X] Confusion Matrix on Test set: ")
    #     p = self.network.predict(self.data_test.X)
    #     m =  pd.crosstab(
    #             pd.Series(self.data_test.y.argmax(axis=1), name='Validation'),
    #             pd.Series(p.argmax(axis=1), name='Prediction')
    #             )
    #     if not self.is_kernel():
    #         print(m)
    #     else:
    #         display(m)

    def learning_graph(self):
        loss = []
        val_loss = []
        metric =[]
        val_metric=[]

        for key in self.history:
            if "loss" in key:
                if "val" in key:
                    val_loss = self.history[key]
                else:
                    loss = self.history[key]
            else:
                if "val" in key:
                    val_metric = self.history[key]
                else:
                    metric = self.history[key]


        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)
        plt.title('Loss Learning Graph')
        plt.plot(loss , c="red", label="Training")
        plt.plot(val_loss, c="green", label="Validation")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()

        plt.subplot(1,2,2)
        plt.title('Metric Learning Graph')
        plt.plot(metric , c="red", label="Training")
        plt.plot(val_metric, c="green", label="Validation")
        plt.ylabel("Metrics")
        plt.xlabel("Epochs")
        plt.legend()

        plt.show()
