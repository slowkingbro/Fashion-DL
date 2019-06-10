
# coding: utf-8

# In[10]:


import os
import numpy as np
from keras.models import load_model
from scipy.io import loadmat


image_width = 224
image_height = 224
channels = 3



folder = "preview/"
train_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f!= '.DS_Store'] # onlyfiles = list of file_name

dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
                     dtype=np.float32)


from keras.models import Model
model = load_model('my_model_0605_1e-07.h5')



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
     file_count_arr[idx-1] = int(file_count[i])

y_train_raw = np.repeat(y_train_raw, file_count_arr, axis = 1)

print()
print('y_labels, y_data_sample (before transpose y_train)',y_train_raw.shape)

X = dataset
X = X/225

y = np.transpose(y_train_raw)

y[y==1] = 0
y[y==2] = 1
y[np.isnan(y)] = 0

print()
print('y_label nan or y = 1 replaced with 0, y = 2 replaces with 1')


# In[22]:


y_pred = model.predict(X)

'''
np.save('y_actual.npy', y)
np.save('y_pred.npy', y_pred)

