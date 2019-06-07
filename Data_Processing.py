
# coding: utf-8

# In[ ]:


import os
import sys

import scipy.io
import scipy.misc
from scipy.misc import toimage
from scipy.io import loadmat
from scipy import ndimage


import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from PIL import Image
#from nst_utils import *
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

#%matplotlib inline

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#from kt_utils import *
#import pydot

K.set_image_data_format('channels_last')

from IPython.display import display
from IPython.display import Image as _Imgdis
from IPython.display import SVG

from time import time
from time import sleep

from sklearn.model_selection import train_test_split


# In[ ]:


folder = "ClothingAttributeDataset/images/"
onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))] # read all files in the 'folder'

#for test purpose
#onlyfiles = onlyfiles[44:45]

#file names in the folder
train_files = []

for _file in onlyfiles:
    train_files.append(_file)
    
print("Files in train_files: %d" % len(train_files))


image_width = 224
image_height = 224
channels = 3

datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')

for _file in train_files:
    img = load_img(folder + "/" + _file)  # this is a PIL image
    img = img.resize((image_height, image_width))
    # Convert to Numpy Array
    x = img_to_array(img)  
    x = x.reshape((1,) + x.shape)
    
    num_image_generated = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix=_file+'_', save_format='jpeg'):
        num_image_generated += 1
        if num_image_generated > 10:
            break # stop the loop after num_image_generated iterations
        
print('%d times data augmentation per image'%num_image_generated)

