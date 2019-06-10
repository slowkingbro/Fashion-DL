
# coding: utf-8

# In[145]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os


# ## Resize loaded image

# In[146]:


image_width = 224
image_height = 224
channels = 3

img_x = load_img('IMG_3144.jpg')  # this is a PIL image
img_x = img_x.resize((image_height, image_width))
x = img_to_array(img_x)
x = np.expand_dims(x, axis=0)


# ## 1. Result from my_model

# In[148]:


from keras.models import Model

model = load_model('my_model_0605_1e-07.h5')
layer_dict = dict([(layer.name, layer) for layer in model.layers])

layer_dict


# In[153]:


# the second last layer
layer_name = 'block5_conv3'

intermediate_layer_my_model = Model(inputs=model.input, 
                                 outputs=model.get_layer(layer_name).output)

intermediate_output_my_model_x = intermediate_layer_my_model.predict(x)


# In[155]:


def cos_sim(a, b):
    a = np.reshape(a, [-1])
    b = np.reshape(b, [-1])
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


## Search the closest from the local dir

folder = "ClothingAttributeDataset/images/"
f_dir = os.listdir(folder)

min_dis = 100000000000
bp_my_model = []


for _file in f_dir[:3]:
    if _file == '.DS_Store':
        continue
#for i in range(1): #comment this out!!!
#    _file = '000681.jpg' #comment this out!!!
    img_db = load_img(folder + "/" +_file)  
    img_db = img_db.resize((image_height, image_width))
    db = img_to_array(img_db)
    db = np.expand_dims(db, axis=0)
    intermediate_output_my_model_db = intermediate_layer_my_model.predict(db)
    dis_my_model = cos_sim(intermediate_output_my_model_db,intermediate_output_my_model_x)#tf.norm(intermediate_output_db-intermediate_output_x, ord='euclidean')


    if dis_db_pred < min_dis:
        bp_my_model = _file
        min_dis = dis_db_pred


print('closet file by my_model is:', bp_my_model)
print(intermediate_output_db)
print(min_dis)


# In[156]:


## 2. Result from VGG

from keras import applications

vgg_model = applications.VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=(224, 224, 3))
#model = vgg_model
layer_dict_vgg = dict([(layer.name, layer) for layer in vgg_model.layers])

intermediate_layer_vgg = Model(inputs=vgg_model.input,  
                                 outputs=vgg_model.get_layer(layer_name).output)

intermediate_output_vgg_x = intermediate_layer_vgg.predict(x)

intermediate_output_vgg.shape


min_dis_vgg = 100000000000
bp_vgg_model = []

for _file in f_dir:
    if _file == '.DS_Store':
        continue
#for i in range(1): #comment this out!!
#    _file = '001697.jpg' #comment this out!!
    img_db = load_img(folder + "/" +_file)   
    img_db = img_db.resize((image_height, image_width))
    db = img_to_array(img_db)
    db = np.expand_dims(db, axis=0)
    intermediate_output_vgg_db = intermediate_layer_vgg.predict(db)
    dis_vgg = cos_sim(intermediate_output_vgg_db, intermediate_output_vgg_x) #flatten and then use cosin similarity


    if dis_db_pred < min_dis_vgg:
        bp_vgg_model = _file
        min_dis_vgg = dis_db_pred

print('closet file by vgg is:', bp_vgg_model)
print(dis_db_pred)


# In[51]:


'''

from keras.layers.convolutional import Deconvolution2D
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K 

conv_out = Deconvolution2D(3, 3, 3, output_shape, border_mode='same')(model.layers[0].output)

deconv_func = K.function([model.input, K.learning_phase()], [conv_out])

test_x = intermediate_output_vgg
X_deconv = deconv_func([test_x, 0 ])

'''


# In[112]:


'''
from keras import Sequential

model = Sequential()
model.add(Deconvolution2D(3, 100, 100, output_shape=(None, 3, 224, 224),
              border_mode='valid',
              input_shape=intermediate_output_vgg[0].shape))
# Note that you will have to change
# the output_shape depending on the backend used.

# we can predict with the model and print the shape of the array.
#x = np.expand_dims(intermediate_output_vgg, axis=0)
#a = tf.transpose(x, [0,3,1,2])
#dummy_input = x
# For TensorFlow dummy_input = np.ones((32, 12, 12, 3))
preds = model.predict(intermediate_output_vgg)
print(preds.shape)

model.layers[0].output

from PIL import Image
import numpy as np


img = Image.fromarray(preds[0], 'RGB')
img.show()
'''

