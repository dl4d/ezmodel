from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
import keras

from ezmodel.ezset import ezset

import math
import random
import matplotlib.pyplot as plt
import numpy as np


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
