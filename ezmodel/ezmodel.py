import pickle
from zipfile import ZipFile
import os
import time
from keras.models import load_model
import sys
import pandas as pd
import matplotlib.pyplot as plt

class ezmodel:

    def __init__(self, load = None, type = None):

        self.data               = None
        self.trainer            = None
        self.type               = None
        self.model_parameters   = None

        if type is not None:
            self.type = type

        if load is not None:
            if type is not None:
                print ("[Fail] ezmodel(): You cannot pass 'load' and 'type' arguments at the same time !")
                return
            else:
                self.load(load)


    def assign(self,data,trainer):
        self.data = data
        self.trainer = trainer


    def train(self,parameters=None):
        #default parameters
        epochs = 10
        callbacks = None
        verbose = 1
        batch_size = 32

        if parameters is not None:
            if "epochs" in parameters:
                epochs = parameters["epochs"]
            if "callbacks" in parameters:
                callbacks = parameters["callbacks"]
            if "verbose" in parameters:
                verbose = parameters["verbose"]
            if "batch_size" in parameters:
                batch_size = parameters["batch_size"]

            self.model_parameters = parameters


        if self.trainer.image_aug is None:
            history = self.trainer.network.fit(
                            self.trainer.X_train,
                            self.trainer.y_train,
                            validation_data=(self.trainer.X_valid,self.trainer.y_valid),
                            epochs=epochs,
                            batch_size = batch_size,
                            callbacks=callbacks,
                            verbose = verbose
                            )

        else:
            print("[X] Training with Data augmentation on Training Set.")
            history = self.trainer.network.fit_generator(
                            self.trainer.image_aug.flow(self.trainer.X_train,self.trainer.y_train,batch_size = batch_size),
                            validation_data = (self.trainer.X_valid,self.trainer.y_valid),
                            steps_per_epoch = self.trainer.X_train.shape[0]//batch_size,
                            epochs=epochs,
                            verbose = verbose
                            )



        #Save history
        if self.trainer.history is None:
            self.trainer.history = history.history
        else:
            for key in history.history:
                self.trainer.history[key] += history.history[key]

        print("\n")



    def evaluate(self):

        print ("[X] Evaluation on Test set: ")
        p = self.trainer.network.evaluate(self.data.X_test,self.data.y_test,verbose=0)

        print ("--- Loss    : ", p[0])
        print ("--- Metrics : ", p[1])
        print("\n")

        return p

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
        p = self.trainer.network.predict(self.trainer.X_valid)
        m =  pd.crosstab(
                pd.Series(self.trainer.y_valid.argmax(axis=1), name='Validation'),
                pd.Series(p.argmax(axis=1), name='Prediction')
                )
        if not self.is_kernel():
            print(m)
        else:
            display(m)

    def learning_graph(self):

        #from IPython.display import display
        #print (self.trainer.history.keys())
        loss = []
        val_loss = []
        metric =[]
        val_metric=[]

        for key in self.trainer.history:
            if "loss" in key:
                if "val" in key:
                    val_loss = self.trainer.history[key]
                else:
                    loss = self.trainer.history[key]
            else:
                if "val" in key:
                    val_metric = self.trainer.history[key]
                else:
                    metric = self.trainer.history[key]

        # if not self.is_kernel():
        #     print("[X] Loss Learning Graph:")
        #     print("Training Loss   : ", loss)
        #     print("Validation Loss : ", val_loss)
        #     print("[X] Metric Learning Graph:")
        #     print("Training Metrics   : ", metric)
        #     print("Validation Metrics : ", val_metric)
        # else:
        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)
        plt.title('Loss Learning Graph')
        plt.plot(loss , c="red", label="Training")
        plt.plot(val_loss, c="green", label="Validation")
        plt.legend()

        plt.subplot(1,2,2)
        plt.title('Metric Learning Graph')
        plt.plot(metric , c="red", label="Training")
        plt.plot(val_metric, c="green", label="Validation")
        plt.legend()

        plt.show()




    def save(self,filename):

        print("[X] Save EZ as :", filename)
        if hasattr(self,"trainer"):
            if hasattr(self.trainer,"network"):
                network = self.trainer.network
                self.trainer.network=None
                network.save(filename+".h5")
                #print("--- EZ trainer has been saved in :", filename,".h5")
            else:
                print("[Notice] No EZ trainer network to save has been found")
        else:
            print("[Notice] No EZ trainer to save has been found")



        filehandler = open(filename+".pkl","wb")
        pickle.dump(self,filehandler)
        filehandler.close()
        print("\n")

        with ZipFile(filename+'.zip', 'w') as myzip:
            myzip.write(filename+".h5")
            myzip.write(filename+".pkl")
        print("--- EZ model has been saved in :", filename,".zip")

        os.remove(filename+".h5")
        os.remove(filename+".pkl")


    def load(self,filename):

        if not os.path.isfile(filename+".zip"):
            raise Exception("[Fail] ezmodel(load) : ", filename,".zip has not been found !")



        zip_ref = ZipFile(filename+".zip", 'r')
        zip_ref.extractall(".")
        zip_ref.close()

        filehandler = open(filename+".pkl", 'rb')
        tmp = pickle.load(filehandler)
        filehandler.close()

        self.data = tmp.data
        self.trainer = tmp.trainer
        self.type = tmp.type
        self.model_parameters = tmp.model_parameters


        self.trainer.network = load_model(filename+".h5")

        os.remove(filename+".h5")
        os.remove(filename+".pkl")

        print("[X]Ezmodel loaded successfully !")
