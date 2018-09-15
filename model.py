
# coding: utf-8

# In[9]:


import csv
#import opencv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, ELU, Flatten, Conv2D, Lambda
from keras.optimizers import Adam
from keras.backend import tf as ktf
from keras.layers.core import Lambda
from keras.regularizers import l2
import cv2
import os
import matplotlib.image as mpimg
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image


# In[10]:


def getdata(path_of_data, flag = False):
    with open(path_of_data + '/driving_log.csv') as csvfile:
        cols = ['Center Image', 'Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Break', 'Speed']
        data = pd.read_csv(csvfile, names = cols, header = 1)
    return data


# In[11]:


data_paths = ['data/data','my_data','reverse_data']
i = 0
data = [0,0,0]
for path in data_paths:
    data_path = path
    data[i] = getdata('../'+path)
    i += 1


# In[12]:


frames = [data[0],data[1],data[2]]
result = pd.concat(frames)
result = result[result["Steering Angle"] != 0]
#print(result)
remove, keep = train_test_split(result, test_size = 0.35)

final_df = [keep, result]
final_df = pd.concat(final_df)
images = final_df[['Center Image', 'Left Image', 'Right Image']]
print(images.shape)
angles = final_df['Steering Angle']
train_images, validation_images, train_angles, validation_angles = train_test_split(images, angles, test_size=0.15, random_state=21)
print(train_images.shape)
print(validation_images.shape)


# In[13]:


def get_image(path, flip=False):
    if path.rfind('/') != -1:
        path = '../data/data/IMG/'+path[path.rfind('/')+1:]
    image = Image.open(path.strip())    
    # flip
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    image = np.array(image, np.float32)
    # Crop image
    image = image[50:130, :]    
    return image


# In[14]:


def generator(images, angles, batch_size = 64,  augment=True):
    batch_img = []
    batch_ang = []
    sample_idx = 0  
    idxs = np.arange(len(images))
    
    while True:
        np.random.shuffle(idxs)
        
        for i in idxs:
            sample_idx = sample_idx + 1
            
            # Center image & steering angle
            batch_img.append(get_image((images.iloc[i]['Center Image'])))
            batch_ang.append(float(angles.iloc[i]))
            
            if augment:
                
                # Left image & adjust steering angle
                batch_img.append(get_image((images.iloc[i]['Left Image'])))
                batch_ang.append(min(1.0, float(angles.iloc[i]) + 0.25))

                # Right image & adjust steering angle
                batch_img.append(get_image((images.iloc[i]['Right Image'])))
                batch_ang.append(max(-1.0, float(angles.iloc[i]) - 0.25))
                
                # Flip image & invert angle
                batch_img.append(get_image((images.iloc[i]['Center Image']), True))
                batch_ang.append((-1.) * float(angles.iloc[i]))
                
            if (sample_idx % len(images)) == 0 or (sample_idx % batch_size) == 0:
                yield np.array(batch_img), np.array(batch_ang)
                batch_img = []
                batch_ang = []


# In[15]:


generator_train = generator(train_images, train_angles)
generator_validation = generator(validation_images, validation_angles, augment=False)
print(generator_train)


# In[8]:


#ConvNet 
model = Sequential()
# Normalize
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(80,320,3)))
# Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
# Add two 3x3 convolution layers (output depth 64, and 64)
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
# Add a flatten layer
model.add(Flatten())
# Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(ELU())
# Add a fully connected output layer
model.add(Dense(1))
# Compile and train the model
model.compile(optimizer=Adam(lr=1e-4), loss='mse')
history = model.fit_generator(generator_train, samples_per_epoch=4*len(train_images), nb_epoch=20,validation_data=generator_validation, nb_val_samples=len(validation_images))


# In[ ]:

print("Save Model")
model.save('model.h5', True)
print("Model Saved")
# In[ ]:

