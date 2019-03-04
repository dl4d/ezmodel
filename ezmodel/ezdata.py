import os
from PIL import Image
import numpy as np
import sys
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


from keras.utils import to_categorical

import keras



class ezdata:
    def __init__(self,parameters=None,load=None):

        self.params      = None

        if parameters is None:
            if load is None:
                #print("[Fail] ezdata() : Please provide a parameter list to instantiate ezdata object !")
                #raise Exception("[Fail] ezdata() : Please provide a parameter list to instantiate ezdata object !")
                pass
                return
            else:
                self.load(load)
                return
        else:
            self.params = parameters

        if "name" not in self.params:
            self.params["name"] = "Noname"

        if "path" not in self.params:
            raise Exception("[Fail] ezdata(): Please provide a path !")
        else:
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


        # if (self.params["type"].lower() == "classification") and (self.params["format"].lower()=="images"):
        #     self.import_classification_images(parameters)
        #     return
        #
        # if (self.params["type"].lower() == "classification") and (self.params["format"].lower()=="table"):
        #     self.import_classification_table(parameters)
        #     return


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
            raise Exception("[Fail] ezdata.import_classification_images() : Path in parameters is not a directory !")

        if not os.path.isdir(parameters["path_mask"]):
            raise Exception("[Fail] ezdata.import_classification_images() : Path Mask in parameters is not a directory !")

        #Images
        print ('[X] Loading Images:', parameters["path"])
        i=0
        for filename in os.listdir(parameters["path"]):
            curimg = os.path.join(parameters["path"], filename)
            img = Image.open(curimg)
            images.append(img)
            image_paths.append(curimg)
            i=i+1
        tot=i
        print ('--- Total images :', str(tot))
        self.images = images
        self.image_paths = image_paths

        #Masks
        print ('[X] Loading Masks:', parameters["path_mask"])
        i=0
        for filename in os.listdir(parameters["path_mask"]):
            curimg = os.path.join(parameters["path_mask"], filename)
            img = Image.open(curimg)
            masks.append(img)
            mask_paths.append(curimg)
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
            raise Exception("[Fail] ezdata.import_classification_table(): Path file in parameters doesn't exist !")

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
                raise Exception("[Fail] ezdata.import_table() : No delimiter suitable for your table have been automatically found. Please provide one using 'table.delim' parameter.")

        if "table.target.column" in parameters:

            Y = table[parameters["table.target.column"]]
            table = table.drop(columns=parameters["table.target.column"])
        else:
            raise Exception("[Fail] ezdata.import_classification_table(): You didn't provide any Target columns into parameters. \n Please assign: 'table.target.column' into parameters list")

        if "table.drop.column" in parameters:
            table = table.drop(columns=parameters["table.drop.column"])

        X = table.values

        self.table      = table
        self.table_path = parameters["path"]
        self.X = X

        #Check the dtype of Y (to know whether it's number or string)
        print ("[X] Table conversion to Keras format: Done")
        print ("--- 'X' and 'y' tensors have been created into current ezdata object.")

        if "table.target.type" in parameters:
            if parameters["table.target.type"]=="string":
                encoder = LabelEncoder()
                Y = encoder.fit_transform(np.squeeze(Y))
                self.synsets = encoder.classes_
                print("--- 'synsets' have been create into current ezdata object.")
        else:
            if Y.dtype == "object":
                encoder = LabelEncoder()
                Y = encoder.fit_transform(np.squeeze(Y))
                self.synsets = encoder.classes_
                print("--- 'synsets' have been create into current ezdata object.")

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
        synsets=[]

        if not os.path.isdir(parameters["path"]):
            raise Exception("[Fail] ezdata.import_classification_images() : Path in parameters is not a directory !")


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
            synsets.append(subdir)
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
        print ("--- 'X' and 'y' tensors have been created into current ezdata object.")
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
        print ("--- 'X' and 'y' tensors have been created into current ezdata object.")
        print("\n")


    def gen_test(self,size=0.2,random_state=42):

        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size=size,random_state=42)

        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test

        print ("[X] Test set generation (size = ,",str(size),",): Done")
        print ("--- Test set : ", self.X_test.shape[0], "images")
        print ("--- 'X_test' and 'y_test' tensors have been created into current ezdata object.")
        print("\n")


    #def preprocess(data,type=None,scaler=None):
    def preprocess(self,X=None,y=None):

        if X.lower() not in ["minmax","standard","mobilenet"]:
            raise Exception('[Fail] preprocess() : Only "minmax","standard","mobilenet" are accepted as preprocessing for X')

        if y is None:
            y="none"

        if y.lower() not in ["minmax","standard","categorical","none"]:
            raise Exception('[Fail] preprocess() : Only "minmax","standard","categorical" or "None" are accepted as preprocessing for Y')

        #X
        if X.lower() == "minmax":
            self.X = self.X.astype("float32")
            self.X,self.scalerX = self.minmax_scaling(self.X)

        if X.lower() == "standard":
            self.X = self.X.astype("float32")
            self.X,self.scalerX = self.standard_scaling(self.X)

        if X.lower() == "mobilenet":
            self.X = self.X.astype("float32")
            self.X = keras.applications.mobilenet.preprocess_input(self.X)
            self.scalerX = "mobilenet"

        if hasattr(self,"X_test"):
            self.X_test = self.X_test.astype("float32")
            if self.scalerX == "mobilenet":
                self.X_test = keras.applications.mobilenet.preprocess_input(self.X_test)
            else:
                self.X_test,_= self.scaler_scaling(self.X_test,self.scalerX)

        #Y
        if y.lower() == "minmax":
            self.y = self.y.astype("float32")
            self.y,self.scalerY = self.minmax_scaling(self.y)
            if hasattr(self,"y_test"):
                self.y_test = self.y_test.astype("float32")
                self.y_test,_= self.scaler_scaling(self.y_test,self.scalerY)

        if y.lower() == "standard":
            self.y = self.y.astype("float32")
            self.y,self.scalerY = self.standard_scaling(self.y)
            if hasattr(self,"y_test"):
                self.y_test = self.y_test.astype("float32")
                self.y_test,_= self.scaler_scaling(self.y_test,self.scalerY)

        if y.lower() == "categorical":
            self.y,self.scalerY = self.categorical_transform(self.y)
            if hasattr(self,"y_test"):
                self.y_test,_       = self.categorical_transform(self.y_test)

        if y.lower() == "none":
            self.scalerY = "none"

        print ("[X] Preprocessing using '",X,"' for X, and '",y,"' for Y.")
        print("\n")




    def minmax_scaling(self,data):
        scalers=[]

        if len(data.shape)==4:
            for i in range(data.shape[3]):
                scaler = MinMaxScaler()
                shape_before = data[:,:,:,i].shape
                a = data[:,:,:,i].reshape(-1,1)
                scalers.append(scaler.fit(a))
                b = scalers[i].transform(a)
                data[:,:,:,i] = b.reshape(shape_before)
                return data,scalers

        if len(data.shape)==2:
            scaler = MinMaxScaler()
            a = data
            scalers.append(scaler.fit(a))
            b = scaler.transform(a)
            data=b
            return data,[scaler]


    def standard_scaling(self,data):

        scalers=[]

        if len(data.shape)==4:
            for i in range(data.shape[3]):
                scaler = StandardScaler()
                shape_before = data[:,:,:,i].shape
                a = data[:,:,:,i].reshape(-1,1)
                scalers.append(scaler.fit(a))
                b = scalers[i].transform(a)
                data[:,:,:,i] = b.reshape(shape_before)
            return data,scalers

        if len(data.shape)==2:
            scaler = StandardScaler()
            a = data
            scalers.append(scaler.fit(a))
            b = scaler.transform(a)
            data=b
            return data,[scaler]


    def categorical_transform(self,data):
        #print ("[X] Preprocessing : Categorical")
        #print("\n")
        return to_categorical(data),"categorical"

    def scaler_scaling(self,data,scaler):

        self.X = self.X.astype("float32")

        for i in range(len(scaler)):
            if len(data.shape)==4:
                shape_before = data[:,:,:,i].shape
                a = data[:,:,:,i].reshape(-1,1)
                b = scaler[i].transform(a)
                data[:,:,:,i] = b.reshape(shape_before)
            if len(data.shape)==2:
                a = data
                b = scaler[i].transform(a)
                data = b
        return data,scaler

    def save(self,filename):
        filehandler = open(filename.strip()+".pkl","wb")
        pickle.dump(self,filehandler)
        filehandler.close()
        print("--- EZ data has been saved in     :",filename,".pkl")
        print("\n")

    def load(self,filename):
        filehandler = open(filename+".pkl", 'rb')
        tmp = pickle.load(filehandler)
        filehandler.close()
        self.__dict__.update(tmp.__dict__)



    def is_kernel(self):
        if 'IPython' not in sys.modules:
            # IPython hasn't been imported, definitely not
            return False
        from IPython import get_ipython
        # check for `kernel` attribute on the IPython instance
        return getattr(get_ipython(), 'kernel', None) is not None

    def show_images(self,n=16):
        #if not isinstance(int(math.sqrt(n)),int):
        #    raise Exception("[Fail] ezdata.show_images(): Please provide n as power of 2 ! (2, 4, 8, 16 ...)")

        if not math.sqrt(n).is_integer():
            raise Exception("[Fail] ezdata.show_images(): Please provide n as a perfect quare ! (2, 4, 9, 16, 25, 36, 49, 64 ...)")
        population = list(range(self.X.shape[0]))
        r = random.sample(population,n)
        fig,axes = plt.subplots(nrows = int(math.sqrt(n)),ncols = int(math.sqrt(n)))
        fig.tight_layout()
        for i in range(n):
            plt.subplot(math.sqrt(n),math.sqrt(n),i+1)

            if (self.X[r[i]].shape[2])==1:
                plt.imshow(np.squeeze(self.X[r[i]]),cmap="gray")
            else:
                plt.imshow(self.X[r[i]])
            #plt.title(self.y[r[i]])
            if hasattr(self,"scalerY"):
                if self.scalerY == "categorical":
                    plt.title(self.synsets[np.argmax(self.y[r[i]])])
                if self.scalerY == "none":
                    plt.title(self.y[r[i]])

            plt.axis("off")

        plt.show()



    def show_table(self,filename=None,head=None):
        from IPython.display import display
        if hasattr(self,"table"):
            if head==None:
                if not self.is_kernel():
                    print(self.table)
                else:
                    display(self.table)
            else:
                if not self.is_kernel():
                    print(self.table.head(head))
                else:
                    display(self.table.head(head))
        else:
            if filename is None:
                raise Exception('[Fail] ezdata.show_table() : Please provide a filename !')
            else:
                #d = pd.read_csv(filename)
                found = False
                for delim in [",",";"," ","\t"]:
                    d = pd.read_csv(filename,delimiter=delim)
                    if d.columns.shape[0] != 1:
                        #print("[Notice] Found a delimiter !")
                        found = True
                        break

                if head==None:
                    if not self.is_kernel():
                        print(d)
                    else:
                        display(d)
                else:
                    if not self.is_kernel():
                        print(d.head(head))
                    else:
                        display(d.head(head))


    def to_npz(self,filename):
        if hasattr(self,"synsets"):
            np.savez(filename+".npz",X=self.X,y=self.y,synsets=self.synsets)
        else:
            np.savez(filename+".npz",X=self.X,y=self.y)

    def from_npz(self,parameters):

        data = np.load(parameters["path"])
        if "X.key" in parameters:
            self.X = data[parameters["X.key"]]
        else:
            self.X = data["X"]

        if "y.key" in parameters:
            self.y = data[parameters["y.key"]]
        else:
            self.y = data["y"]
        if "synsets.key" in parameters:
            self.synsets = data[parameters["synsets.key"]]
        else:
            if "synsets" in data:
                self.synsets = data["synsets"]

        if "name" in parameters:
            self.name = parameters["name"]
        else:
            self.name = "NoName"
        print("[X] Loading from NPZ !")
