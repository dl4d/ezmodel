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
import copy



def split(data,size=0.2,random_state=42):

    #Version 2.0 : VIRTUAL IMAGE SET
    if hasattr(data,"virtual"):
        if data.virtual == True:
            return split_virtual(data,size,random_state)

    X_train,X_test,y_train,y_test = train_test_split(data.X,data.y,test_size=size,random_state=42)

    print ("[X] Test set generation (size = ,",str(size),",): Done")
    print ("--- Train set : ", X_train.shape[0], "samples")
    print ("--- Test set : ", X_test.shape[0], "samples")
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

#Version 2.0 : VIRTUAL IMAGE SET
def split_virtual(data,size,random_state):


    data.imagedg._validation_split=size
    print("[Notice: ezutils.split_virtual(): Unused argument 'random_state']")

    train = ezset()
    train.virtual=True
    train.params = data.params
    train.imagedg = data.imagedg
    train.generator = train.imagedg.flow_from_directory(
            train.params["path"],
            target_size=train.params["resize"],
            batch_size=train.params["batch_size"],
            color_mode=train.params["color_mode"],
            class_mode=train.params["class_mode"],
            shuffle=True,
            subset="training")
    #Create virtual entry from memory (no memory consumption)
    train.X = np.zeros((train.generator.samples,) + train.generator.image_shape)
    train.y = np.zeros((train.generator.samples,) + (train.generator.num_classes,))


    test = ezset()
    test.virtual=True
    test.params = data.params
    test.imagedg = data.imagedg
    test.generator = test.imagedg.flow_from_directory(
            test.params["path"],
            target_size=test.params["resize"],
            batch_size=test.params["batch_size"],
            color_mode=test.params["color_mode"],
            class_mode=test.params["class_mode"],
            shuffle=True,
            subset="validation")
    #Create virtual entry from memory (no memory consumption)
    test.X = np.zeros((test.generator.samples,) + test.generator.image_shape)
    test.y = np.zeros((test.generator.samples,) + (test.generator.num_classes,))


    return train,test


def binarize(data,class0=None,class0_label="class0",class1=None,class1_label="class1"):
    if class0 is None:
        raise Exception('[Fail] ezutils.keep(): Please pass a set of class0 name as parameters !')

    if class1 is None:
        raise Exception('[Fail] ezutils.keep(): Please pass a set of class0 name as parameters !')

    if not hasattr(data,"synsets"):
        raise Exception('[Fail] ezutils.keep(): Please pass dataset containing synsets !')

    @np.vectorize
    def contained0(x):
        return x in S0
    @np.vectorize
    def contained1(x):
        return x in S1

    labels0 = [key  for (key, value) in data.synsets.items() if value in class0]
    S0 = set(labels0)
    indix0 = np.where(contained0(data.y))[0].astype('int64')
    data.y[indix0]=0

    labels1 = [key  for (key, value) in data.synsets.items() if value in class1]
    S1 = set(labels1)
    indix1 = np.where(contained1(data.y))[0].astype('int64')
    data.y[indix1]=1

    newdict = dict()
    newdict[0] = class0_label
    newdict[1] = class1_label

    indices = np.concatenate((indix0,indix1))
    data.X = data.X[indices]
    data.y = data.y[indices]
    data.synsets = newdict

    return data


def keep(data,classes=None):

    if classes is None:
        raise Exception('[Fail] ezutils.keep(): Please pass a set of class name as parameters !')

    if not hasattr(data,"synsets"):
        raise Exception('[Fail] ezutils.keep(): Please pass dataset containing synsets !')

    newdict=dict()
    i=0
    indices =[]
    for c in classes:
        class0 = [key  for (key, value) in data.synsets.items() if value == c]
        indix = np.where(data.y==class0[0])[0]
        indices = np.concatenate((indices,indix))
        indices = indices.astype('int64')
        data.y[indix] = i
        newdict[i] = data.synsets[class0[0]]
        i=i+1
    data.X = data.X[indices]
    data.y = data.y[indices]
    data.synsets = newdict
    return data



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
            #RGB data
            if data.X[r[i]].dtype == "float32" or data.X[r[i]].dtype == "float64":
                plt.imshow(data.X[r[i]].astype('int32'))
            elif data.X[r[i]].dtype == "int32" or data.X[r[i]].dtype == "int64":
                plt.imshow(data.X[r[i]])
            else:
                raise Exception('[Fail] ezutils.show_images(): only float32, float64, int32, int64 are supported type ')

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
