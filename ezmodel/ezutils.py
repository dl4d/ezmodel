from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
import keras
from keras.models import load_model

from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel
from ezmodel.ezlosses import *

import math
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from zipfile import ZipFile



def split(data,size=0.2,random_state=42):

    X_train,X_test,y_train,y_test = train_test_split(data.X,data.y,test_size=size,random_state=42)

    print ("[X] Test set generation (size = ,",str(size),",): Done")
    print ("--- Train set : ", X_train.shape[0], "images")
    print ("--- Test set : ", X_test.shape[0], "images")
    print("\n")

    train = ezset()
    train.X = X_train
    train.y = y_train
    train.synsets = data.synsets

    test = ezset()
    test.X = X_test
    test.y = y_test
    test.synsets = data.synsets

    return train,test

def show_images(data,n=16):

    #Checkers
    if not math.sqrt(n).is_integer():
        raise Exception("\n\n\t[Fail] ezutils.show_images(): Please provide n as a perfect quare ! (2, 4, 9, 16, 25, 36, 49, 64 ...)")

    if len(data.X.shape)==2:
        raise Exception("\n\n\t[Fail] ezutils.show_images(): Your input doesn't seem to be an Images: tensor dim should be 4, your provided ",str(len(data.X.shape))," instead ... please use an image instead, or check your preprocessing (flatten ?) !")

    population = list(range(data.X.shape[0]))
    r = random.sample(population,n)
    fig,axes = plt.subplots(nrows = int(math.sqrt(n)),ncols = int(math.sqrt(n)))
    fig.tight_layout()
    mask = False
    for i in range(n):
        plt.subplot(math.sqrt(n),math.sqrt(n),i+1)

        if (data.X[r[i]].shape[2])==1:
            plt.imshow(np.squeeze(data.X[r[i]]),cmap="gray")
        else:
            plt.imshow(data.X[r[i]])

        if data.synsets is not None:
            if len(data.y.shape)==1:
                plt.title(data.synsets[data.y[r[i]]])
            elif len(data.y.shape)==2:
                plt.title(data.synsets[np.argmax(data.y[r[i]])])
        else:
            if len(data.y.shape)==4:
                mask = True

    if mask==True:
        fig,axes = plt.subplots(nrows = int(math.sqrt(n)),ncols = int(math.sqrt(n)))
        fig.tight_layout()
        for i in range(n):
            plt.subplot(math.sqrt(n),math.sqrt(n),i+1)

            if (data.y[r[i]].shape[2])==1:
                plt.imshow(np.squeeze(data.y[r[i]]),cmap="gray")
            else:
                plt.imshow(data.y[r[i]])
            plt.axis("off")

    plt.axis("off")
    plt.show()


def load_ezmodel(filename):
        if not os.path.isfile(filename+".zip"):
            raise Exception("[Fail] ezmodel(load) : ", filename,".zip has not been found !")

        ez = ezmodel(empty=True)

        zip_ref = ZipFile(filename+".zip", 'r')
        zip_ref.extractall(".")
        zip_ref.close()

        filehandler = open(filename+".data.pkl", 'rb')
        tmp = pickle.load(filehandler)
        filehandler.close()
        ez.data_train = tmp[0]
        ez.data_test  = tmp[1]

        filehandler = open(filename+".trans.pkl", 'rb')
        tmp = pickle.load(filehandler)
        filehandler.close()
        ez.transformerX = tmp[0]
        ez.transformerY  = tmp[1]

        filehandler = open(filename+".params.pkl", 'rb')
        tmp = pickle.load(filehandler)
        filehandler.close()
        ez.model_parameters = tmp

        filehandler = open(filename+".hist.pkl", 'rb')
        tmp = pickle.load(filehandler)
        filehandler.close()
        ez.history = tmp

        filehandler = open(filename+".aug.pkl", 'rb')
        tmp = pickle.load(filehandler)
        filehandler.close()
        ez.augmentation = tmp

        #Network + optimizer + custom losses
        import ezmodel.ezlosses as tmp
        import types
        l=[getattr(tmp, a) for a in dir(tmp)
          if isinstance(getattr(tmp, a), types.FunctionType)]
        values = l
        keys = []
        for i in l : keys.append((i.__name__))
        custom_objects = dict(zip(keys,values))
        ez.network = load_model(filename+".h5",custom_objects=custom_objects)


        os.remove(filename+".h5")
        os.remove(filename+".data.pkl")
        os.remove(filename+".trans.pkl")
        os.remove(filename+".params.pkl")
        os.remove(filename+".hist.pkl")
        os.remove(filename+".aug.pkl")

        print("[X]Ezmodel loaded successfully !")

        return ez
