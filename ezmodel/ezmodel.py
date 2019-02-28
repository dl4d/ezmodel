import pickle
from zipfile import ZipFile
import os

class ezmodel:

    def __init__(self, load = None, type = None):

        data    = None
        trainer = None
        type    = None

        if type is not None:
            self.type = type


    def assign(self,data,trainer):
        self.data = data
        self.trainer = trainer


    def train(self,parameters=None):
        #default parameters
        epochs = 10
        callbacks = None
        verbose = 1

        if parameters is not None:
            if "epochs" in parameters:
                epochs = parameters["epochs"]
            if "callbacks" in parameters:
                callbacks = parameters["callbacks"]
            if "verbose" in parameters:
                verbose = parameters["verbose"]

        history = self.trainer.network.fit(
                        self.trainer.X_train,
                        self.trainer.y_train,
                        validation_data=(self.trainer.X_valid,self.trainer.y_valid),
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose = verbose
                        )

        print("\n")



    def evaluate(self):

        print ("[X] Evaluation on Test set: ")
        p = self.trainer.network.evaluate(self.data.X_test,self.data.y_test,verbose=0)

        print ("--- Loss    : ", p[0])
        print ("--- Metrics : ", p[1])
        print("\n")

        return p


    def save(self,filename):

        print("[X] Save EZ as :", filename)
        if hasattr(self,"trainer"):
            if hasattr(self.trainer,"network"):
                network = self.trainer.network
                self.trainer.network=None
                network.save(filename+".h5")
                print("--- EZ trainer has been saved in :", filename,".h5")
            else:
                print("[Notice] No EZ trainer network to save has been found")
        else:
            print("[Notice] No EZ trainer to save has been found")


        filehandler = open(filename+".pkl","wb")
        pickle.dump(self,filehandler)
        print("--- EZ data has been saved in     :",filename,".pkl")
        print("\n")

        with ZipFile(filename+'.zip', 'w') as myzip:
            myzip.write(filename+".h5")
            myzip.write(filename+".pkl")

        os.remove(filename+".h5")
        os.remove(filename+".pkl")
