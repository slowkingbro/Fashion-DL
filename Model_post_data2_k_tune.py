
# coding: utf-8

# In[13]:


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


K.set_image_data_format('channels_last')

from IPython.display import display
from IPython.display import Image as _Imgdis
from IPython.display import SVG

from time import time
from time import sleep

from sklearn.model_selection import train_test_split

import tensorflow as tf


# In[39]:


import datetime

now = datetime.datetime.now()

dt_now = now.strftime("%Y-%m-%d")
dt_now


# In[7]:


image_width = 224
image_height = 224
channels = 3


# In[8]:


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
    if i % 500 == 0:
        print("%d images to array" % i)

print("All images to array!")
print("train_data_shape",dataset.shape)


# In[9]:


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


# In[10]:


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

# In[11]:


X = dataset
X = X/225

y = np.transpose(y_train_raw)

re_label = label_ls

'''
### delete this for complete label!!! 
re_label = ['black_GT','blue_GT', 'brown_GT','cyan_GT', 'gender_GT', 'gray_GT', 'green_GT', 'purple_GT', 'red_GT','white_GT', 'yellow_GT']
print('num_relabel='+str(len(re_label)))
y = y[:,[x in re_label for x in label_ls]]
###
'''



y[y==1] = 0
y[y==2] = 1
y[np.isnan(y)] = 0

print()
print('y_label nan or y = 1 replaced with 0, y = 2 replaces with 1')

len_y = y.shape[1]

# ## 5. Build model

# In[26]:


from keras import applications
from keras.models import Model
from keras import optimizers
from sklearn.metrics import f1_score



loss_track = []

# fit a model and plot learning curve
def fit_model(X, y, lrate, k1):
    
    def cus_loss(y_true, y_pred, k=k1):
        result = []
        total_loss = 0
        
        for i in range(y_pred.shape[1]):    
            col = y_pred[:,i]
            col_true = y_true[:,i]
            #if label_ls[i]=='gender_GT':
            #    col_true*=4
            total_loss += -1*(k1 * col_true * tf.math.log(col) + (1-col_true) * tf.math.log(1-col))
        total_loss = tf.reduce_mean(total_loss)
        return total_loss

    vgg_model = applications.VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=(224, 224, 3))

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

    # Getting output tensor of the last VGG layer that we want to include
    x = vgg_model.output
    x = Flatten()(x)
    x = Dense(len_y, activation='sigmoid')(x)

    custom_model = Model(input=vgg_model.input, output=x)

    # Make sure that the pre-trained bottom layers are not trainable
    for layer in custom_model.layers[:12]:
        layer.trainable = False

    opt = optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    
    custom_model.compile(loss = cus_loss,
                         optimizer = opt,
                         metrics = ['accuracy'])
    
    history = custom_model.fit(x=X, y=y, validation_split=0.33, batch_size=32, epochs=5)
    print()
    print('###INFO:my_model_'+dt_now+'_'+str(lrate)+'_'+str(k1)+'num_label_'+str(len(re_label))+'.h5')
    y_pred = custom_model.predict(X)
    
    y_pred = y_pred.flatten()
    y_pred[y_pred>=0.5]=1
    y_pred[y_pred<0.5]=0
    y_true = y.flatten()

    print('f1_score ('+str(lrate)+'&'+str(k1)+')='+str(f1_score(y_true, y_pred)))
    custom_model.save('my_model_'+dt_now+'_'+str(lrate)+'_'+str(k1)+'num_label_'+str(len(re_label))+'.h5')
                      
    # plot learning curves
    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='test')
    pyplot.title('param_k='+str(k1))
 
    loss_track.append(history.history['loss'][-1])


 ### reate learning curves for different learning rates
learning_rates = [1E-6,1E-7,1E-8]
K = [1.5,2,3,4,5,10]


for i in range(len(learning_rates)):
    for j in range(len(K)):
    	print('param_k=',K[j])
    	print('learning_rate=',learning_rates[i])
    # determine the plot number
    	plot_no = 420 + (i+1)
    	pyplot.subplot(plot_no)
    # fit model and plot learning curves for a learning rate
    	fit_model(X, y, learning_rates[i],K[j])

    
# show learning curves
pyplot.savefig('learning_rate_'+dt_now+'.pdf')


# In[21]:


print()
print('label list is:',re_label)
print()


print('loss_track:',loss_track)
print('finished!!!')



