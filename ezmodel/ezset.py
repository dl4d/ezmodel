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
from collections import Counter
import copy
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from keras.utils import to_categorical

import keras

#from ezmodel import ezutils


class ezset:

    def __init__(self,parameters=None,virtual=False):

        self.X = None
        self.y = None
        self.synsets = None
        self.params = None
        self.virtual = False

        if parameters is None:
            return
        if "path" not in parameters:
            raise Exception("ezset.init() : Please provide at least a path into parameters")

        self.params = parameters


        #Version 2.0 : VIRTUAL IMAGE SET
        if virtual:
            self.init_virtual()
            return

        if (os.path.isdir(self.params["path"])):
            if "path_mask" not in self.params:
                if "index" not in self.params:
                    #Images classification from directory
                    self.import_classification_images(parameters)
                    return
                else:
                    #Image classification from directory + Index file
                    self.import_classification_images_from_indexes(parameters)
                    return
            else:
                #Image segmentation from images/ masks/ directories
                self.import_segmentation_images(parameters)

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

        #Case of Joker for semgentation
        if "***" in self.params["path"]:
            self.import_segmentation_images2(parameters)
            return


        raise Exception('[Fail]: Path Not found !')

    #Version 2.0 : VIRTUAL IMAGE SET
    def init_virtual(self):

        if "target_size" not in self.params:
            self.params["target_size"]=None
        if "batch_size" not in self.params:
            self.params["batch_size"]=None
        if "color_mode" not in self.params:
            self.params["color_mode"]="RGB"
        if "class_mode" not in self.params:
            self.params["class_mode"]="categorical"

        self.virtual = True
        self.imagedg = keras.preprocessing.image.ImageDataGenerator(
                rescale = self.params["scaling"]
        )
        self.generator = self.imagedg.flow_from_directory(
                self.params["path"],
                target_size = self.params["resize"],
                batch_size = self.params["batch_size"],
                color_mode = self.params["color_mode"],
                class_mode = self.params["class_mode"],
                shuffle=True
        )


        #Create virtual entry from memory (no memory consumption)
        self.X = np.zeros((self.generator.samples,) + self.generator.image_shape)
        self.y = np.zeros((self.generator.samples,) + (self.generator.num_classes,))




    def import_classification_images_from_indexes(self,parameters):

        #Read the index and find a delimiter
        found = False
        for delim in [",",";"," ","\t"]:
            table = pd.read_csv(parameters["index"],delimiter=delim)
            if table.columns.shape[0] != 1:
                print("[Notice] Found a delimiter !")
                found = True
                break
        if not found:
            raise Exception("[Fail] ezset.import_classification_images_from_indexes() : No delimiter suitable for your table have been automatically found. Please provide one using 'table.delim' parameter.")

        if "image.path.column" not in parameters:
            raise Exception("[Fail] ezset.import_classification_images_from_indexes() : No 'image.path.column' has been defined into parameters. Please define it !")

        if "target.column" not in parameters:
            raise Exception("[Fail] ezset.import_classification_images_from_indexes() : No 'target.column' has been defined into parameters. Please define it !")

        images =[]
        image_paths=[]
        labels =[]
        synsets={}

        #Images from index file
        print ('[X] Loading Images:', parameters["path"])
        print ('--- from index file:', parameters["path"])
        # for i in range(table.shape[0]):
        #     curimg = os.path.join(parameters["path"], table.loc[i,parameters["image.path.column"]])
        #     print(curimg)

        #Labels
        l = table[parameters["target.column"]]
        if l.dtype == "object":
            encoder = LabelEncoder()
            l = encoder.fit_transform(np.squeeze(l))
            self.synsets = {v: k for v, k in enumerate(encoder.classes_)}
            print("--- 'synsets' have been created into current ezset object.")

        #Images
        i=0
        for filename in sorted(os.listdir(parameters["path"])):
            if (i%100)==0:
                print(str(i) + "/" + str(len(l)))

            curimg = os.path.join(parameters["path"], filename)
            #print(filename)
            img = Image.open(curimg)
            imgcopy = img.copy()
            images.append(imgcopy)
            image_paths.append(curimg)
            img.close()
            i=i+1

            search = os.path.splitext(filename)[0]
            w = np.where(table[parameters["image.path.column"]]==search)[0]
            labels.append(l[w])

        tot=i
        print ('--- Total images :', str(tot))

        self.images = images
        self.image_paths = image_paths
        self.index = parameters["index"]
        self.labels = labels

        print("\n")

        if "resize" in self.params:
          self.images_to_keras(self.params["resize"])
        else:
          self.images_to_keras()

        return


    def import_segmentation_images2(self,parameters):

        #Collapsing masks
        if "collapse" not in parameters:
            collapse = True
        else:
            collapse = parameters["collapse"]

        parent_dir = parameters["path"].split("***")[0]
        image_dir  = parameters["path"].split("***")[1][1:]
        mask_dir   = parameters["path_mask"].split("***")[1][1:]

        ids = next(os.walk(parent_dir))[1]

        #images = np.zeros((len(ids),parameters["resize"][0],parameters["resize"][1],3),dtype=np.uint8)
        #masks  = np.zeros((len(ids),parameters["resize"][0],parameters["resize"][1],1),dtype=np.uint8)
        images = []
        image_paths = []
        masks  = []
        mask_paths = []

        print ('[X] Loading Images & Masks from :', parent_dir)

        for n,directories in tqdm(enumerate(sorted(os.listdir(parent_dir))),total=len(ids)):

            #Images
            curimdir = os.path.join(parent_dir,directories,image_dir)
            for image in sorted(os.listdir(curimdir)):
                curimg = os.path.join(curimdir,image)
                img = Image.open(curimg)
                imgcopy = img.copy()
                images.append(imgcopy)
                image_paths.append(curimg)
                img.close()

            #Masks
            curmaskdir = os.path.join(parent_dir,directories,mask_dir)
            tmp=[]
            for mask in sorted(os.listdir(curmaskdir)):
                curmask = os.path.join(curmaskdir,mask)
                img = Image.open(curmask)
                imgcopy = img.copy()
                tmp.append(imgcopy)
                img.close()

            #mask collapsing ?
            if collapse:
                a=np.array(tmp[0])
                for i in tmp[1:]:
                    a += np.array(i)
            else:
                a=tmp
            img = Image.fromarray(a)
            imgcopy = img.copy()
            masks.append(imgcopy)
            img.close()


            # print(len(tmp))
        self.images = images
        self.image_paths = image_paths
        self.masks  = masks

        print ('--- Total images :', len(self.images))
        print ('--- Total masks  :', len(self.masks))

        if "resize" in self.params:
          self.images_masks_to_keras(self.params["resize"])
        else:
          self.images_masks_to_keras()


    # def import_segmentation_images2(self,parameters):
    #     """
    #     Import segmentation image:
    #     Top directory -> Samples -> Images and Masks
    #     """
    #     #Collapsing masks
    #     if "collapse" not in parameters:
    #         collapse = True
    #     else:
    #         collapse = parameters["collapse"]
    #
    #
    #     images_pd  = parameters["path"].split("***")[0]
    #     images_dir = parameters["path"].split("***")[1][1:]
    #     masks_pd   = parameters["path_mask"].split("***")[0]
    #     masks_dir  = parameters["path_mask"].split("***")[1][1:]
    #
    #     images = []
    #     image_paths=[]
    #     masks = []
    #     mask_paths = []
    #
    #     #Images
    #     print ('[X] Loading Images:', parameters["path"])
    #     i=0
    #     for directories in sorted(os.listdir(images_pd)):
    #         pi = os.path.join(images_pd,directories,images_dir)
    #         for image in sorted(os.listdir(pi)):
    #             # print(image)
    #             curimg = os.path.join(pi,image)
    #
    #             img = Image.open(curimg)
    #             imgcopy = img.copy()
    #             images.append(imgcopy)
    #             image_paths.append(curimg)
    #             img.close()
    #
    #             i=i+1
    #     print ('--- Total images :', str(i))
    #     self.images = images
    #     self.image_paths = image_paths
    #
    #     #Masks
    #     print ('[X] Loading Masks:', parameters["path_mask"])
    #     i=0
    #     for directories in sorted(os.listdir(masks_pd)):
    #         pm = os.path.join(masks_pd,directories,masks_dir)
    #         k=0
    #         local_masks= []
    #         local_mask_paths = []
    #         for image in sorted(os.listdir(pm)):
    #             # print(image)
    #             curimg = os.path.join(pm,image)
    #
    #             img = Image.open(curimg)
    #             imgcopy = img.copy()
    #             # masks.append(imgcopy)
    #             # mask_paths.append(curimg)
    #             local_masks.append(imgcopy)
    #             local_mask_paths.append(curimg)
    #             img.close()
    #
    #             i=i+1
    #             k=k+1
    #
    #         #Collapsing
    #         if (k>1) and (collapse):
    #             im = np.array(local_masks[0])
    #             for m in local_masks[1:]:
    #                 im+=np.array(m)
    #             masks.append(Image.fromarray(im))
    #             mask_paths.append(local_mask_paths)
    #
    #     print ('--- Total Masks :', str(i))
    #     if collapse:
    #         print ('--- Total Masks collapsed :', str(len(masks)))
    #     self.masks = masks
    #     self.mask_paths = mask_paths
    #
    #     print("\n")
    #
    #     if "resize" in self.params:
    #       self.images_masks_to_keras(self.params["resize"])
    #     else:
    #       self.images_masks_to_keras()



        #sys.exit()



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
            # img = Image.open(curimg)
            # images.append(img)
            # image_paths.append(curimg)
            # img.close()

            img = Image.open(curimg)
            imgcopy = img.copy()
            images.append(imgcopy)
            image_paths.append(curimg)
            img.close()


            i=i+1
        tot=i
        print ('--- Total images :', str(tot))
        self.images = images
        self.image_paths = image_paths

        #Masks
        print ('[X] Loading Masks:', parameters["path_mask"])
        i=0
        for filename in sorted(os.listdir(parameters["path_mask"])):
            curimg = os.path.join(parameters["path_mask"], filename)
            # img = Image.open(curimg)
            # masks.append(img)
            # mask_paths.append(curimg)
            # img.close()
            img = Image.open(curimg)
            imgcopy = img.copy()
            masks.append(imgcopy)
            mask_paths.append(curimg)
            img.close()


            i=i+1
        tot=i
        print ('--- Total images :', str(tot))
        self.masks = masks
        self.mask_paths = mask_paths

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


        print ('[X] Loading directory:', parameters["path"])

        k=0
        tot=0
        for subdir in sorted(os.listdir(parameters["path"])):
            curdir = os.path.join(parameters["path"],subdir)
            i=0
            for filename in sorted(os.listdir(curdir)):
                curimg = os.path.join(curdir, filename)
                # img = Image.open(curimg)
                # images.append(img)
                # labels.append(k)
                # image_paths.append(curimg)
                # img.close()
                img = Image.open(curimg)
                imgcopy = img.copy()
                images.append(imgcopy)
                labels.append(k)
                image_paths.append(curimg)
                img.close()


                i=i+1
            #synsets.append(subdir)
            synsets[k]=subdir
            k=k+1
            tot=tot+i
            print ('--- subdir: ', subdir, '(',str(i),' images )')
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
        #squeeze because npz add a singleton dimension once saved
        if len(self.y.shape)==2:
            if self.y.shape[1]==1:
                self.y = np.squeeze(self.y)

        if "synsets.key" in parameters:
            self.synsets = data[parameters["synsets.key"]][()]
        else:
            if "synsets" in data:
                self.synsets = data["synsets"][()]
        if "name" in parameters:
            self.name = parameters["name"]
        else:
            self.name = "NoName"
        print("[X] Loading from : " + parameters["path"])
        print ("--- 'X' and 'y' tensors have been created into current ezset object.")
        print("\n")

    def to_npz(self,filename):

        print("[X] Saving ezset to npz : ", filename)
        if hasattr(self,"synsets"):
            if hasattr(self,"name"):
                np.savez(filename,X=self.X,y=self.y,synsets=self.synsets,name=self.name)
            else:
                np.savez(filename,X=self.X,y=self.y,synsets=self.synsets)
        else:
            if hasattr(self,"name"):
                np.savez(filename,X=self.X,y=self.y,name=self.name)
            else:
                np.savez(filename,X=self.X,y=self.y)
        print('--- Done !')

    def transform(self,X=None,y=None):

        if self.virtual:
            return self.transform_virtual(X,y)

        if X is not None:
            if X.lower()=="minmax":
                transformerX = to_minmax(self.X)
            elif X.lower()=="standard":
                transformerX = to_standard(self.X)
            elif X.lower()=="mobilenet":
                transformerX = to_mobilenet(self.X)
            elif X.lower()=="mobilenetv2":
                transformerX = to_mobilenetv2(self.X)
            elif X.lower()=="xception":
                transformerX = to_xception(self.X)
            elif X.lower()=="inceptionv3":
                transformerX = to_inceptionv3(self.X)
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


    def transform_virtual(self,X,y):
            if X.lower()=="minmax":
                self.transformerX = X.lower()
            elif X.lower()=="standard":
                self.transformerX = X.lower()
            else:
                raise Exception("ezset.transform(): Unknown transformer X : ", X, " in ezset 'virtual' context ")
            if y.lower()=="minmax":
                self.transformery = y.lower()
            elif X.lower()=="standard":
                self.transformery = y.lower()
            else:
                raise Exception("ezset.transform(): Unknown transformer y : ", y, " in ezset 'virtual' context ")


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
        print("[X] Flatten: Done")

    def falseRGB(self):
        self.X = np.concatenate((self.X,self.X,self.X), axis=3)

    def autoencoder(self):
        self.orig_y = np.copy(self.y)
        self.y = np.copy(self.X)

    def undersampling(self,min):
        #squeeze because npz add a singleton dimension once saved
        if len(self.y.shape)==2:
            if self.y.shape[1]==1:
                self.y = np.squeeze(self.y)

        c = Counter(self.y)
        u = np.unique(self.y)
        indices = []
        for i in range(int(u.max()+1)):
            indix = np.where(self.y==i)[0].tolist()
            if c[i]>=min:
                r = random.sample(indix,min)
            else:
                r = indix
            indices+=r

        self.X = self.X[indices]
        self.y = self.y[indices]
        print("[X] Undersampling with min=",min,": Done")


    def input(self):
        return self.X.shape[1:]

    def output(self,transformers=None):
        #Temporary transform data
        if transformers is not None:
            input0 = copy.deepcopy(self)
            input0.preprocess(X=transformers[0],y=transformers[1])
        else:
            input0 = self

        if len(input0.y.shape)==2:
            return input0.y.shape[1]
        if len(input0.y.shape)==1:
            return 1
        if len(input0.y.shape)==4:
            return input0.y.shape







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

#mobilenetv2
def to_mobilenetv2(data):
    return [keras.applications.mobilenet_v2.preprocess_input]

#xception
def to_xception(data):
    return [keras.applications.xception.preprocess_input]

#inceptionv3
def to_inceptionv3(data):
    return [keras.applications.inception_v3.preprocess_input]


#vgg16
def to_vgg16(data):
    return [keras.applications.vgg16.preprocess_input]

#vgg16
def to_vgg19(data):
    return [keras.applications.vgg19.preprocess_input]
