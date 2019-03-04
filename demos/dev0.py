import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

from ezmodel.ezmodel   import ezmodel
from ezmodel.ezdata    import ezdata
from ezmodel.eztrainer import eztrainer


ezd = ezdata()
ezd.X = ezd.image_from_url(
                    url="https://www.almanac.com/sites/default/files/styles/primary_image_in_article/public/image_nodes/strawberries-1.jpg",
                    img_size=(128,128)
                    )
ezd.y = None

import pickle
from urllib.request import urlopen
ezd.synsets = pickle.load(urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl') )

ezt = eztrainer()
ezt.network = ezt.PretrainedNetwork("mobilenet",ezd.X.shape[1:])
ezt.network.summary()

ezd.preprocess(X="mobilenet",y=None)

import numpy as np
p = ezt.network.predict(ezd.X)
top = 5
ind_five = (-p).argsort()[-3:][::-1][0][0:top]
for i in range(top):
    print(ezd.synsets[ind_five[i]]," : ", np.around(p[0][ind_five[i]]*100,3),"%")
