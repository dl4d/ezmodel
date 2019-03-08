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

    def __init__(self,train=None,test=None,network=None,optimizer=None,transformers=None,augmentation=None,empty=False):

        if not empty:
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
#            self.optimizer = optimizer
            self.transformerX = transformers[0]
            self.transformerY = transformers[1]
            self.model_parameters = None
            self.history = None
            self.augmentation = None

            self.network.compile(**optimizer)

            if augmentation is not None:
                self.keras_augmentation(augmentation)
        else:
            self.data_train   = None
            self.data_test    = None
            self.network = None
#            self.optimizer = None
            self.transformerX = None
            self.transformerY = None
            self.model_parameters = None
            self.history = None
            self.augmentation = None



    def train(self,parameters=None):
        #default parameters
        epochs = 10
        callbacks = None
        verbose = 1
        batch_size = 32
        validation_data = None


        # Transformers
        print("[X] Transformers : ")
        train = copy.deepcopy(self.data_train)
        train.preprocess(X=self.transformerX,y=self.transformerY)
        print("--- Use transformers to preprocess Training set : Done");
        if "validation_split" not in parameters:
            test = copy.deepcopy(self.data_test)
            test.preprocess(X=self.transformerX,y=self.transformerY)
            print("--- Use transformers to preprocess Test set : Done");


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
                X_train,X_valid,y_train,y_valid = train_test_split(train.X,train.y,test_size=parameters["validation_split"],random_state=42)
                validation_data = (X_valid,y_valid)
            else:
                print("[Notice] Test set will be used ad Validation set for training !")
                validation_data = (test.X,test.y)

            self.model_parameters = parameters

        if self.augmentation is None:
            history = self.network.fit(
                            train.X,
                            train.y,
                            validation_data=validation_data,
                            epochs=epochs,
                            batch_size = batch_size,
                            callbacks=callbacks,
                            verbose = verbose
                            )
        else:
            print("[X] Training with Data augmentation on Training Set.")
            history = self.network.fit_generator(
                            self.augmentation.flow(train.X,train.y,batch_size = batch_size),
                            validation_data = validation_data,
                            steps_per_epoch = train.X.shape[0]//batch_size,
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
        print("Loss: ",self.network.loss,":",p[0])
        k=0
        for m in self.network.metrics:
            print("Metrics: ",m,":",p[1+k])
            k=k+1
        return p

    def predict(self):
        print ("[X] Prediction on Test set:")
        test = copy.deepcopy(self.data_test)
        test.preprocess(X=self.transformerX,y=self.transformerY)
        print("--- Use transformers to preprocess Test set : Done")
        p = self.network.predict(test.X,verbose=0)
        return p

    def keras_augmentation(self,parameters):
        image_gen = ImageDataGenerator(**parameters)
        if self.data_train.X is None:
            raise Exception("[Fail] ezmodel.augmentation(): No Training set has been added to this ezmodel object")
        image_gen.fit(self.data_train.X, augment=True)
        self.augmentation = image_gen
        print("[X] Keras ImageDataGenerator has been added to ezmodel")
        print("\n")
    # def keras_augmentation(self,parameters):
    #
    #     # Transformers
    #     print("[X] Transformers : ")
    #     train = copy.deepcopy(self.data_train)
    #     train.preprocess(X=self.transformerX,y=self.transformerY)
    #     print("--- Use transformers to preprocess Training set : Done");
    #     if "validation_split" not in parameters:
    #         test = copy.deepcopy(self.data_test)
    #         test.preprocess(X=self.transformerX,y=self.transformerY)
    #         print("--- Use transformers to preprocess Test set : Done");
    #
    #     image_gen = ImageDataGenerator(**parameters)
    #     mask_gen  = ImageDataGenerator(**parameters)
    #     seed = 1
    #
    #     if self.data_train.X is None:
    #         raise Exception("[Fail] ezmodel.augmentation(): No Training set has been added to this ezmodel object")
    #
    #     image_gen.fit(self.data_train.X, augment=True, seed=seed)
    #     mask_gen.fit(self.data_train.y, augment=True, seed=seed)
    #
    #     image_generator = image_gen.flow()
    #
    #
    #
    #     self.augmentation = image_gen
    #     print("[X] Keras ImageDataGenerator has been added to ezmodel")
    #     print("\n")




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

    def is_kernel(self):
        if 'IPython' not in sys.modules:
            # IPython hasn't been imported, definitely not
            return False
        from IPython import get_ipython
        # check for `kernel` attribute on the IPython instance
        return getattr(get_ipython(), 'kernel', None) is not None


    def confusion_matrix(self):
        from IPython.display import display
        print("[X] Confusion Matrix on Test set: ")
        #p = self.trainer.network.predict(self.trainer.X_valid)
        p = self.predict()

        test = copy.deepcopy(self.data_test)
        test.preprocess(X=None,y=self.transformerY)

        if self.data_test.synsets is not None:
            m =  pd.crosstab(
                     #pd.Series(self.trainer.y_valid.argmax(axis=1), name='Validation'),
                     pd.Series(test.y.argmax(axis=1), name='Validation'),
                     pd.Series(p.argmax(axis=1), name='Prediction')
                     )
            if not self.is_kernel():
                print(m)
            else:
                display(m)
        else:
            raise Exception('ezmodel.confusion_matrix(): Impossible to display confusion matrix: No synsets found into test dataset! ')

    def save(self,filename):

        print("[X] Save EZmodel as :", filename)

        #Network
        if self.network is not None:
            self.network.save(filename+".h5")
        else:
            print("[Notice] No Network to save has been found !")

        #Data
        filehandler = open(filename+".data.pkl","wb")
        pickle.dump((self.data_train,self.data_test),filehandler)
        filehandler.close()

        #Transformers
        filehandler = open(filename+".trans.pkl","wb")
        pickle.dump((self.transformerX,self.transformerY),filehandler)
        filehandler.close()

        filehandler = open(filename+".params.pkl","wb")
        pickle.dump(self.model_parameters,filehandler)
        filehandler.close()

        filehandler = open(filename+".hist.pkl","wb")
        pickle.dump(self.history,filehandler)
        filehandler.close()

        filehandler = open(filename+".aug.pkl","wb")
        pickle.dump(self.augmentation,filehandler)
        filehandler.close()

        with ZipFile(filename+'.zip', 'w') as myzip:
            myzip.write(filename+".h5")
            myzip.write(filename+".data.pkl")
            myzip.write(filename+".trans.pkl")
            myzip.write(filename+".params.pkl")
            myzip.write(filename+".hist.pkl")
            myzip.write(filename+".aug.pkl")

        print("--- EZ model has been saved in :", filename,".zip")

        os.remove(filename+".h5")
        os.remove(filename+".data.pkl")
        os.remove(filename+".trans.pkl")
        os.remove(filename+".params.pkl")
        os.remove(filename+".hist.pkl")
        os.remove(filename+".aug.pkl")
