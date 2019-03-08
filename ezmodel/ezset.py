import os
from PIL import Image
import numpy as np
import sys
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
import math
import requests
from io import BytesIO
from urllib.request import urlopen

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from keras.utils import to_categorical

import keras

#from ezmodel import ezutils


class ezset:

    def __init__(self,parameters=None):

        self.X = None
        self.y = None
        self.synsets = None
        self.params = None

        if parameters is None:
            return
        if "path" not in parameters:
            raise Exception("ezset.init() : Please provide at least a path into parameters")
        else:
            self.params = parameters
            #Images classification from directory
            if (os.path.isdir(self.params["path"])) and ("path_mask" not in self.params):
                self.import_classification_images(parameters)
                #self.type = "classification"
                return
            #Images segmentation from directory
            if (os.path.isdir(self.params["path"])) and (os.path.isdir(self.params["path_mask"])):
                self.import_segmentation_images(parameters)
                #self.type = "segmentation"
                return


            if (os.path.isfile(self.params["path"])):

                extension = os.path.splitext(self.params["path"])[1]
                if extension == ".csv":
                    #Table Classification or regression
                    self.import_table(parameters)
                    return
                if extension == ".npz":
                    self.from_npz(self.params)
                    return
                raise Exception('File extension/type not recognized ! Should be "csv" or "npz" !')
                return

            #Image classification from directory and csv index file
            if (os.path.isdir(self.params["path"])) and (os.path.isfile(self.params["path_index"])):
                self.import_classification_images_from_indexes()
                #self.type="classification"


    def import_classification_images_from_index(self,parameters):
        print('TODO: import_classification_images_from_index()')
        return

    #SEGMENTATION IMAGES from DIRECTORY
    def import_segmentation_images(self,parameters):

        images =[]
        image_paths=[]
        masks =[]
        mask_paths=[]

        if not os.path.isdir(parameters["path"]):
            raise Exception("[Fail] ezset.import_classification_images() : Path in parameters is not a directory !")

        if not os.path.isdir(parameters["path_mask"]):
            raise Exception("[Fail] ezset.import_classification_images() : Path Mask in parameters is not a directory !")

        #Images
        print ('[X] Loading Images:', parameters["path"])
        i=0
        for filename in sorted(os.listdir(parameters["path"])):
            curimg = os.path.join(parameters["path"], filename)
            img = Image.open(curimg)
            images.append(img)
            image_paths.append(curimg)
            i=i+1
        tot=i
        print ('--- Total images :', str(tot))
        self.images = images
        self.image_paths = image_paths
        print(image_paths)

        #Masks
        print ('[X] Loading Masks:', parameters["path_mask"])
        i=0
        for filename in sorted(os.listdir(parameters["path_mask"])):
            curimg = os.path.join(parameters["path_mask"], filename)
            img = Image.open(curimg)
            masks.append(img)
            mask_paths.append(curimg)
            i=i+1
        tot=i
        print ('--- Total images :', str(tot))
        self.masks = masks
        self.mask_paths = mask_paths
        print(image_paths)


        print("\n")

        if "resize" in self.params:
          self.images_masks_to_keras(self.params["resize"])
        else:
          self.images_masks_to_keras()


    #CLASSIFICATION IMAGES from CSVTABLE annuaire

    #CLASSIFICATION - TABLES
    def import_table(self,parameters):

        table =[]
        table_path = []

        if not os.path.isfile(parameters["path"]):
            raise Exception("[Fail] ezset.import_classification_table(): Path file in parameters doesn't exist !")

        print ('[X] Loading :', parameters["path"])

        #Delimiter
        if "table.delimiter" in parameters:
            table = pd.read_csv(parameters["path"],delimiter=parameters["table.delimiter"])
        else:
            found = False
            for delim in [",",";"," ","\t"]:
                table = pd.read_csv(parameters["path"],delimiter=delim)
                if table.columns.shape[0] != 1:
                    print("[Notice] Found a delimiter !")
                    found = True
                    break
            if not found:
                raise Exception("[Fail] ezset.import_table() : No delimiter suitable for your table have been automatically found. Please provide one using 'table.delim' parameter.")

        if "table.target.column" in parameters:

            Y = table[parameters["table.target.column"]]
            table = table.drop(columns=parameters["table.target.column"])
        else:
            raise Exception("[Fail] ezset.import_classification_table(): You didn't provide any Target columns into parameters. \n Please assign: 'table.target.column' into parameters list")

        if "table.drop.column" in parameters:
            table = table.drop(columns=parameters["table.drop.column"])

        X = table.values

        self.table      = table
        self.table_path = parameters["path"]
        self.X = X

        #Check the dtype of Y (to know whether it's number or string)
        print ("[X] Table conversion to Keras format: Done")
        print ("--- 'X' and 'y' tensors have been created into current ezset object.")

        if "table.target.type" in parameters:
            if parameters["table.target.type"]=="string":
                encoder = LabelEncoder()
                Y = encoder.fit_ezsetm(np.squeeze(Y))
                #self.synsets = encoder.classes_
                self.synsets = {v: k for v, k in enumerate(encoder.classes_)}
                print("--- 'synsets' have been create into current ezset object.")
        else:
            if Y.dtype == "object":
                encoder = LabelEncoder()
                Y = encoder.fit_transform(np.squeeze(Y))
                #self.synsets = encoder.classes_
                self.synsets = {v: k for v, k in enumerate(encoder.classes_)}
                print("--- 'synsets' have been create into current ezset object.")

        # if "table.target.type" in parameters:
        #     if parameters["table.target.type"]=="string":
        #         encoder = LabelEncoder()
        #         Y = encoder.fit_transform(np.squeeze(Y))
        #         self.synsets = encoder.classes_

        self.y = Y



        print("\n")



    # CLASSIFICATION - IMAGES
    def import_classification_images(self,parameters):
        images =[]
        labels =[]
        image_paths=[]
        synsets={}

        if not os.path.isdir(parameters["path"]):
            raise Exception("[Fail] ezset.import_classification_images() : Path in parameters is not a directory !")


        print ('[X] Loading :', parameters["path"])

        k=0
        tot=0
        for subdir in os.listdir(parameters["path"]):
            curdir = os.path.join(parameters["path"],subdir)
            i=0
            for filename in os.listdir(curdir):
                curimg = os.path.join(curdir, filename)
                img = Image.open(curimg)
                images.append(img)
                labels.append(k)
                image_paths.append(curimg)
                i=i+1
            #synsets.append(subdir)
            synsets[k]=subdir
            k=k+1
            tot=tot+i
            print ('--- dir: ', subdir, '(',str(i),' images )')
        print ('--- Total images :', str(tot))
        self.images = images
        self.labels = labels
        self.image_paths = image_paths
        self.synsets = synsets
        print('--- Synsets have been generated')
        print("\n")

        if "resize" in self.params:
          self.images_to_keras(self.params["resize"])
        else:
          self.images_to_keras()



    def images_masks_to_keras(self,resize=None):

        #Images
        im=[]
        for image in self.images:
            r = image
            if resize is not None:
                r = image.resize((resize[0],resize[1]), Image.NEAREST)
            im.append(r)
        imgarray=list();
        for i in range(len(im)):
            tmp = np.array(im[i])
            imgarray.append(tmp)
        imgarray = np.asarray(imgarray)

        if len(imgarray.shape)==1:
            raise Exception("[Fail] images_to_keras() : Image size heterogeneity !  Size of images into the dataset are not the same. You should try to use 'resize' parameters to make them homogenous.")
        if len(imgarray.shape)==3:
            imgarray = np.expand_dims(imgarray,axis=3)
        self.X = imgarray.astype('float32')

        #Masks
        im=[]
        for image in self.masks:
            r = image
            if resize is not None:
                r = image.resize((resize[0],resize[1]), Image.NEAREST)
            im.append(r)
        imgarray=list();
        for i in range(len(im)):
            tmp = np.array(im[i])
            imgarray.append(tmp)
        imgarray = np.asarray(imgarray)

        if len(imgarray.shape)==1:
            raise Exception("[Fail] images_to_keras() : Masks size heterogeneity !  Size of Masks into the dataset are not the same. You should try to use 'resize' parameters to make them homogenous.")
        if len(imgarray.shape)==3:
            imgarray = np.expand_dims(imgarray,axis=3)
        self.y = imgarray.astype('float32')


        print ("[X] Images & Masks conversion to Keras format: Done")
        print("\n")


    def images_to_keras(self,resize=None):

        im=[]
        for image in self.images:
            r = image
            if resize is not None:
                r = image.resize((resize[0],resize[1]), Image.NEAREST)
            im.append(r)
        imgarray=list();
        for i in range(len(im)):
            tmp = np.array(im[i])
            imgarray.append(tmp)
        imgarray = np.asarray(imgarray)

        if len(imgarray.shape)==1:
            raise Exception("[Fail] images_to_keras() : Image size heterogeneity !  Size of images into the dataset are not the same. You should try to use 'resize' parameters to make them homogenous.")


        if len(imgarray.shape)==3:
            imgarray = np.expand_dims(imgarray,axis=3)


        self.X = imgarray.astype('float32')
        self.y = np.asarray(self.labels)

        print ("[X] Images conversion to Keras format: Done")
        print("\n")

    def from_npz(self,parameters):
        data = np.load(parameters["path"])
        if "X.key" in parameters:
            self.X = data[parameters["X.key"]].astype('float32')
        else:
            self.X = data["X"].astype('float32')

        if "y.key" in parameters:
            self.y = data[parameters["y.key"]].astype('float32')
        else:
            self.y = data["y"].astype('float32')
        if "synsets.key" in parameters:
            self.synsets = data[parameters["synsets.key"]]
        else:
            if "synsets" in data:
                self.synsets = data["synsets"]

        if "name" in parameters:
            self.name = parameters["name"]
        else:
            self.name = "NoName"
        print("[X] Loading from : " + parameters["path"])
        print("\n")

    def transform(self,X=None,y=None):
        if X is not None:
            if X.lower()=="minmax":
                transformerX = to_minmax(self.X)
            elif X.lower()=="standard":
                transformerX = to_standard(self.X)
            elif X.lower()=="mobilenet":
                transformerX = to_mobilenet(self.X)
            elif X.lower()=="vgg16":
                transformerX = to_vgg16(self.X)
            elif X.lower()=="vgg19":
                transformerX = to_vgg19(self.X)
            else:
                raise Exception("ezset.transform(): Unknown transformer X: ", X)
        else:
            transformerX = None

        if y is not None:
            if y.lower()=="minmax":
                transformerY = to_minmax(self.y)
            elif y.lower()=="standard":
                transformerY = to_standard(self.y)
            elif y.lower()=="categorical":
                transformerY = to_categorical(self.y)
            elif y.lower()=="labelencoder":
                transformerY = to_labelencoder(self.y)
            else:
                raise Exception("ezset.transform(): Unknown transformer y: ", y)
        else:
            transformerY = None

        #Transform
        #self.preprocess(X=transformerX,y=transformerY)



        return (transformerX,transformerY)


    def preprocess(self,X=None,y=None):

        if X is not None:
            if (len(X)==1) and (X[0].__name__ != "transform"):
                self.X = X[0](self.X)
            elif (len(X)==1) and (X[0].__name__ == "transform") and (len(self.X.shape)==2):
                self.X = X[0](self.X)
            else:
                i=0
                #if len(self.X.shape)==4:
                for scalerX in X:
                    before_shape = self.X[:,:,:,i].shape
                    a = scalerX(self.X[:,:,:,i].reshape(-1,1))
                    b = a.reshape(before_shape)
                    self.X[:,:,:,i] = b
                    i=i+1

        if y is not None:
            if (len(y)==1) and (y[0].__name__ != "transform"):
                self.y = y[0](self.y)
            elif (len(y)==1) and (y[0].__name__ == "transform") and (len(self.X.shape)==2):
                self.y = y[0](self.y)
            else:
                i=0
                for scalerY in y:
                    before_shape = self.y[:,:,:,i].shape
                    a = scalerY(self.y[:,:,:,i].reshape(-1,1))
                    b = a.reshape(before_shape)
                    self.y[:,:,:,i] = b
                    i=i+1

    def flatten(self):
        self.X = self.X.reshape(-1,self.X.shape[1]*self.X.shape[2]*self.X.shape[3])



#Minmax scaler
def to_minmax(data):
    scalers=[]
    if len(data.shape)==4:
        for i in range(data.shape[3]):
            scaler = MinMaxScaler()
            a = data[:,:,:,i].reshape(-1,1)
            b = scaler.fit(a)
            scalers.append(b.transform)
        return scalers
    if len(data.shape)==2:
        scaler = MinMaxScaler()
        a = data
        b = scaler.fit(a)
        scalers.append(b.transform)
    return scalers

#Standard scaler
def to_standard(data):
    scalers=[]
    if len(data.shape)==4:
        for i in range(data.shape[3]):
            scaler = StandardScaler()
            a = data[:,:,:,i].reshape(-1,1)
            b = scaler.fit(a)
            scalers.append(b.transform)
        return scalers
    if len(data.shape)==2:
        scalers=[]
        scaler = StandardScaler()
        a = data
        b = scaler.fit(a)
        scalers.append(b.transform)
    return scalers

#LabelEncoder
def to_labelencoder(data):
    scalers =[]
    scaler = LabelEncoder()
    b = scaler.fit(data)
    scalers.append(b.transform)
    return scalers

#categorical_transform
def to_categorical(data):
    return [keras.utils.to_categorical]

#mobilenet
def to_mobilenet(data):
    return [keras.applications.mobilenet.preprocess_input]

#vgg16
def to_vgg16(data):
    return [keras.applications.vgg16.preprocess_input]

#vgg16
def to_vgg19(data):
    return [keras.applications.vgg19.preprocess_input]
