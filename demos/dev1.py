import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

#from ezmodel.ezmodel import ezmodel
from ezmodel.ezset import ezset
from ezmodel.ezmodel import ezmodel

from ezmodel.ezutils import split,show_images,load_ezmodel
from ezmodel.eznetwork import Pretrained
import keras


import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.filters import prewitt
from skimage.filters import threshold_otsu
import time


# [EZSET]  -------------------------------------------------------------------
parameters={
    "name"      : "Blob Test",
    "path"      : "C:\\Users\\daian\\Desktop\\DATA\\Blob_test\\",
    "resize"    : (64,64)
}
data = ezset(parameters)

print(data.X.mean())


# [EZMODEL]  ----------------------------------------------------------------
ez = load_ezmodel(filename="blob")

print(ez.data_test.X.max())
print(ez.data_test.X.mean())
import sys
sys.exit()


ez.data_test = data
ez.transformerY = None

p =ez.predict()

print(p.max())

#print(p.shape)

#import sys
#sys.exit()

#ez.evaluate()



for i in range(p.shape[0]):
    im = np.squeeze(p[i])
    plt.imshow(im)
    plt.show()

import sys
sys.exit





for i in range(ez.data_test.X.shape[0]):
    im = np.squeeze(p[i]>0.5).astype('uint8')
    im2 = prewitt(im)
    threshold_global_otsu = threshold_otsu(im2)
    im3 = im2 <= threshold_global_otsu
    im3=im3.astype('float32')
    plt.imshow(im3*np.squeeze(ez.data_test.X[i]),cmap="gray")
    plt.show()
    time.sleep(0.5)



#
# print(ez.data_test.X.max())
#
#
# A[:,:,0] = im3
# A[:,:,1] = np.squeeze(ez.data_test.X[0]/255)
# A[:,:,2] = np.squeeze(ez.data_test.X[0]/255)
# A=A.astype("uint8")
# print(A)
#
# plt.imshow(A)
# plt.show()



# print(sum(im))
# im2, contours, hierarchy = cv2.findContours(im.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(im, contours, -1, (0,255,0), 3)
# cv2.imshow("image",im)
# # Wait indefinitely until you push a key.  Once you do, close the windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# for i in range(p.shape[0]):
#     plt.imshow(np.squeeze((p[i]<0.5)*ez.data_test.X[i]),cmap="gray")
#     plt.show()
#
#print(ez.data_test.X.max())
