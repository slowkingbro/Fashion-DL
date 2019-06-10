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

#image_name = 'IMG_3144.jpg'
image_name='image3.jpg'
print(image_name)


img_x = load_img(image_name)  # this is a PIL image
img_x = img_x.resize((image_height, image_width))
x = img_to_array(img_x)
x = np.expand_dims(x, axis=0)


# ## 1. Result from my_model

# In[153]:

from keras.models import Model
import tensorflow as tf

param_K = 10
print('param_=k'+str(param_K))

def cus_loss(y_true, y_pred, k1=param_K):
    result = []
    total_loss = 0

    for i in range(y_pred.shape[1]):
        col = y_pred[:,i]
        col_true = y_true[:,i]
        total_loss += -1*(k1 * col_true * tf.math.log(col) + (1-col_true) * tf.math.log(1-col))
    total_loss = tf.reduce_mean(total_loss)
    return total_loss

model_name = 'my_model_2019-06-09_1e-07_3num_label_20.h5'
#model_name = 'my_model_2019-06-09_1e-0710num_label(11.h5'
#model_name = 'my_model_2019-06-09_1e-075num_label(11_gender*4.h5'
#model_name = 'my_model_2019-06-09_1e-075num_label(11_gender*2.h5'
#model_name = 'my_model_2019-06-09_1e-075num_label(11).h5'
#model_name = 'my_model_2019-06-09_1e-075num_label(20).h5'
#model_name = 'my_model_2019-06-09_1e-072num_label(11).h5'
#model_name = 'my_model_2019-06-08_1e-0710.h5'
#model_name = 'my_model_2019-06-08_1e-073.h5'
#model_name='my_model_2019-06-08_1e-071.5.h5'
#model = load_model('my_model_0605_1e-07.h5')
model = load_model(model_name, custom_objects={'cus_loss': cus_loss})
layer_dict = dict([(layer.name, layer) for layer in model.layers])

layer_dict
                                   
  
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

### heapify
import heapq

li = [(0,'na'), (0,'na'), (0,'na'), (0,'na'), (0,'na')]
heapq.heapify(li)

###

## Search the closest from the local dir

folder = "ClothingAttributeDataset/images/"
f_dir = os.listdir(folder)


for _file in f_dir:
    if _file == '.DS_Store':
        continue

    img_db = load_img(folder + "/" +_file)
    img_db = img_db.resize((image_height, image_width))
    db = img_to_array(img_db)
    db = np.expand_dims(db, axis=0)
    intermediate_output_my_model_db = intermediate_layer_my_model.predict(db)
    dis_my_model = cos_sim(intermediate_output_my_model_db,intermediate_output_my_model_x)
    db_tuple = (dis_my_model, _file)
    heapq.heappush(li,db_tuple)
    heapq.heappop(li)



print()
print('the model is: ',model_name)
print('closest files by my_model are:', li)
print('my_model_prediction for test img is=',model.predict(x))
print()

# In[156]:


## 2. Result from VGG

from keras import applications

vgg_model = applications.VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=(224, 224, 3))
                                                                                                                                                                                                                   136,1         45%

  #model = vgg_model
layer_dict_vgg = dict([(layer.name, layer) for layer in vgg_model.layers])

intermediate_layer_vgg = Model(inputs=vgg_model.input,
                                 outputs=vgg_model.get_layer(layer_name).output)

intermediate_output_vgg_x = intermediate_layer_vgg.predict(x)


li_vgg = [(0,'na'), (0,'na'), (0,'na'), (0,'na'), (0,'na')]
heapq.heapify(li_vgg)


for _file in f_dir:
    if _file == '.DS_Store':
        continue

    img_db = load_img(folder + "/" +_file)
    img_db = img_db.resize((image_height, image_width))
    db = img_to_array(img_db)
    db = np.expand_dims(db, axis=0)
    intermediate_output_vgg_db = intermediate_layer_vgg.predict(db)
    dis_vgg = cos_sim(intermediate_output_vgg_db, intermediate_output_vgg_x) #flatten and then use cosin similarity
    db_tuple = (dis_vgg, _file)
    heapq.heappush(li_vgg,db_tuple)
    heapq.heappop(li_vgg)


print()
print('the model is original vgg')
print('closest files by my_model are:', li_vgg)
#print('vgg_model_prediction for test img is=',vgg_model.predict(x))
print()


