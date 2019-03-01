import os
from PIL import Image
import numpy as np
import sys
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


from keras.utils import to_categorical



class ezdata:
    def __init__(self,parameters=None,load=None):

        self.params      = None

        if parameters is None:
            if load is None:
                #print("[Fail] ezdata() : Please provide a parameter list to instantiate ezdata object !")
                raise Error("[Fail] ezdata() : Please provide a parameter list to instantiate ezdata object !")
            else:
                self.load(load)
                return
        else:
            self.params = parameters

        if (self.params["type"].lower() == "classification") and (self.params["format"].lower()=="images"):
            self.import_classification_images(parameters)
            return

        if (self.params["type"].lower() == "classification") and (self.params["format"].lower()=="table"):
            self.import_classification_table(parameters)
            return

    #CLASSIFICATION - TABLES
    def import_classification_table(self,parameters):

        table =[]
        table_path = []

        if not os.path.isfile(parameters["path"]):
            print("[Fail] ezdata.import_classification_table(): Path file in parameters doesn't exist !")
            sys.exit()

        print ('[X] Loading :', parameters["path"])

        table = pd.read_csv(parameters["path"])

        if "table_target_column" in parameters:
            Y = table[parameters["table_target_column"]]
            table = table.drop(columns=parameters["table_target_column"])
        else:
            print("[Fail] ezdata.import_classification_table(): You didn't provide any Target columns into parameters. \n Please assign: 'table_target_colum' into parameters list")
            sys.exit()

        if "table_drop_column" in parameters:
            table = table.drop(columns=parameters["table_drop_column"])

        X = table.values

        self.table      = table
        self.table_path = parameters["path"]
        self.X = X

        if "table_target_column_type" in parameters:
            if parameters["table_target_column_type"]=="string":
                encoder = LabelEncoder()
                Y = encoder.fit_transform(np.squeeze(Y))
                self.synsets = encoder.classes_

        self.y = Y

        print ("[X] Table conversion to Keras format: Done")
        print ("--- 'X' and 'y' tensors have been created into current ezdata object.")
        print("\n")



    # CLASSIFICATION - IMAGES
    def import_classification_images(self,parameters):

        images =[]
        labels =[]
        image_paths=[]
        synsets=[]

        if not os.path.isdir(parameters["path"]):
            print("[Fail] ezdata.import_classification_images() : Path in parameters is not a directory !")
            sys.exit()

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
        print("\n")

        if "resize" in self.params:
            self.to_keras(self.params["resize"])
        else:
            self.images_to_keras()


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
            print ("[Fail] images_to_keras() : Image size heterogeneity !  Size of images into the dataset are not the same. You should try to use 'resize' parameters to make them homogenous.")
            sys.exit()

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

        if X.lower() not in ["minmax","standard"]:
            print('[Fail] preprocess() : Only "minmax","standard" are accepted as preprocessing for X')
            sys.exit()

        if y.lower() not in ["minmax","standard","categorical"]:
            print('[Fail] preprocess() : Only "minmax","standard","categorical" are accepted as preprocessing for Y')
            sys.exit()

        #X
        if X.lower() == "minmax":
            self.X,self.scalerX = self.minmax_scaling(self.X)

        if X.lower() == "standard":
            self.X,self.scalerX = self.standard_scaling(self.X)

        self.X_test,_= self.scaler_scaling(self.X_test,self.scalerX)

        #Y
        if y.lower() == "minmax":
            self.y,self.scalerY = self.minmax_scaling(self.y)
            self.y_test,_= self.scaler_scaling(self.y_test,self.scalerY)

        if y.lower() == "standard":
            self.y,self.scalerY = self.standard_scaling(self.y)
            self.y_test,_= self.scaler_scaling(self.y_test,self.scalerY)

        if y.lower() == "categorical":
            self.y,self.scalerY = self.categorical_transform(self.y)
            self.y_test,_       = self.categorical_transform(self.y_test)

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
