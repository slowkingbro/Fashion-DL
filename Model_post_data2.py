
# coding: utf-8

# In[48]:


import os
import sys

import scipy.io
import scipy.misc
from scipy.misc import toimage
from scipy.io import loadmat
from scipy import ndimage


import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib import pyplot

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


# In[24]:


image_width = 224
image_height = 224
channels = 3


# In[25]:


folder = "preview/"
train_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f!= '.DS_Store'] # onlyfiles = list of file_name

dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
                     dtype=np.float32)

i=0
for _file in train_files:
    img = load_img(folder + "/" + _file)  # this is a PIL image
    img = img.resize((image_height, image_width))
    x = img_to_array(img)  
    dataset[i] = x
    i += 1
    if i % 250 == 0:
        print("%d images to array" % i)

print("All images to array!")
print("train_data_shape",dataset.shape)


# In[26]:


folder = "ClothingAttributeDataset/labels/"
label_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f!= '.DS_Store']
y_train = []
label_ls = []

for _file in label_files:
    x = loadmat(folder + _file)
    x = x['GT'].repeat(1)
    y_train.append(x)
    file_name = _file[0:_file.find('.mat')]
    label_ls.append(file_name)

y_train_raw = np.asarray(y_train)

print()
print('label_list:', label_ls)
print('y_train_raw.shape', y_train_raw.shape)


# In[27]:


#count data augmentation frequency
folder = "preview/"
f_preview = os.listdir(folder)
file_count = {}

for _file in f_preview:
    if _file == '.DS_Store':
        continue
    _file = _file[0:10]
    if _file in file_count:
        file_count[_file]+=1
    else:
        file_count[_file]=1

file_count_arr = np.zeros([1856],dtype=int)

for i in file_count:
    idx = int(i[:6])
    #print(idx)
    file_count_arr[idx-1] = int(file_count[i])
    #print(file_count[i])

y_train_raw = np.repeat(y_train_raw, file_count_arr, axis = 1)

print()
print('y_labels, y_data_sample (before transpose y_train)',y_train_raw.shape)


# ## 4. Convert X, y and train test data split

# In[28]:


X = dataset
X = X/225

y = np.transpose(y_train_raw)

y[y==1] = 0
y[y==2] = 1
y[np.isnan(y)] = 0

print()
print('y_label nan or y = 1 replaced with 0, y = 2 replaces with 1')


# ## 5. Build model

# In[53]:


from keras import applications
from keras.models import Model
from keras import optimizers


# fit a model and plot learning curve
def fit_model(X, y, lrate):
    
    vgg_model = applications.VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=(224, 224, 3))

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

    # Getting output tensor of the last VGG layer that we want to include
    x = vgg_model.output

    '''
    # Stacking a new simple convolutional network on top of it    
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    '''
    x = Flatten()(x)
    x = Dense(20, activation='sigmoid')(x)
    custom_model = Model(input=vgg_model.input, output=x)

    # Make sure that the pre-trained bottom layers are not trainable
    for layer in custom_model.layers[:12]:
        layer.trainable = False

    opt = optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    custom_model.compile(loss = 'binary_crossentropy',
                         optimizer = opt,
                         metrics = ['accuracy'])
    
    history = custom_model.fit(x=X, y=y, validation_split=0.33, batch_size=32, epochs=18)
    
    # plot learning curves
    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='test')
    pyplot.title('lrate='+str(lrate), pad=-50)
 
 
# create learning curves for different learning rates
learning_rates = [1E-0, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7]

for i in range(len(learning_rates)):
    # determine the plot number
    plot_no = 420 + (i+1)
    pyplot.subplot(plot_no)
    # fit model and plot learning curves for a learning rate
    fit_model(X, y, learning_rates[i])
# show learning curves
pyplot.show()
fig.savefig('learning_rate.pdf')


# In[20]:


# list all data in history
print(history.history.keys())
print()
print('train_accuracy:',history.history['acc'])
print('test_accuracy:',history.history['val_acc'])
print('train_loss:',history.history['loss'])
print('test_loss:',history.history['val_loss'])


# In[21]:


print()
print('label list is:',label_ls)
print()

custom_model.save('my_model_0605.h5')
print('finished!!!')

