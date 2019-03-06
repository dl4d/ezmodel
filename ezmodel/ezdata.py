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



class ezdata:

    def __init__(self,parameters=None):
