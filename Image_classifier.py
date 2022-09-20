# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:47:31 2022

@author: mahon
"""

from matplotlib import pyplot as plt
import os
import random

#Lets start by checking out our image data. Begin with cats -- meow
_, _, cat_images = next(os.walk(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Cat'))

fig, ax = plt.subplots(3, 3, figsize=(20,10))

for idx, image in enumerate(random.sample(cat_images,9)):
    img_read = plt.imread('C:/Users/mahon/Documents/Python Scripts/NN Practice/PetImages/Cat/' + image)
    ax[int(idx/3), idx%3].imshow(img_read)
    ax[int(idx/3), idx%3].axis('off')
    ax[int(idx/3), idx%3].set_title('Cat/'+image)
    
plt.show()

#And now lets look at dogs -- woof!
_, _, dog_images = next(os.walk(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Dog'))

fig, ax = plt.subplots(3, 3, figsize=(20,10))

for idx, image in enumerate(random.sample(cat_images,9)):
    img_read = plt.imread('C:/Users/mahon/Documents/Python Scripts/NN Practice/PetImages/Dog/'+image)
    ax[int(idx/3), idx%3].imshow(img_read)
    ax[int(idx/3), idx%3].axis('off')
    ax[int(idx/3), idx%3].set_title('Cat/'+image)
    
plt.show()

#Alright, so we can read in images. But they are quite big. We'll have to manage
#our dataset using some of Keras' utilities. 

#First we need to augment our data (images). Flip, roatate, etc so the classifier generalizes better.
from keras.preprocessing.image import ImageDataGenerator 

image_generator = ImageDataGenerator(rotation_range = 30,
                                     width_shift_range = .20,
                                     height_shift_range =.20,
                                     zoom_range =.20,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

fig, ax = plt.subplots(2,3, figsize=(20,10) )
all_images=[]

#Lets grab a random cat and check out this image augmentor
_, _, cat_images = next(os.walk(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Cat'))
random_image=random.sample(cat_images,1)[0]
random_image=plt.imread('C:/Users/mahon/Documents/Python Scripts/NN Practice/PetImages/Cat/' + random_image)
all_images.append(random_image)

random_image=random_image.reshape((1,)+random_image.shape)
sample_augmented_images = image_generator.flow(random_image)

for _ in range(5):
    augmented_images=sample_augmented_images.next()
    for image in augmented_images:
        all_images.append(image.astype('uint8'))



for idx, image in enumerate(all_images):
    ax[int(idx/3), idx%3].imshow(image)
    ax[int(idx/3), idx%3].axis('off')
    if idx==0:
        ax[int(idx/3), idx%3].set_title('OG Image')
    else:
        ax[int(idx/3), idx%3].set_title('Augmented Image {}'.format(idx))

plt.show()

#%% Okay lets build some directories with train/test data. Use the handy shutil class
import os
import shutil

#Start with Cats
new_path=r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Train\Cat\\'
old_path=r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Cat\\'

files = os.listdir(old_path)
upper=int(len(files))

#Put 80% in a training directory
for i in range(2,upper):
    if i<.8*upper:
        old_path=r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Cat\\'+str(i)+'.jpg'
        shutil.move(old_path, new_path)
    
    else:
        new_path=r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Test\Cat\\'
        old_path=r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Cat\\'+str(i)+'.jpg'
        shutil.move(old_path, new_path)


#Now lets grab dogs
new_path=r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Train\Dog\\'
old_path=r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Dog\\'

files = os.listdir(old_path)
upper=int(len(files))

#Put 80% in a training directory
for i in range(upper):
    if i<.8*upper:
        old_path=r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Dog\\'+str(i)+'.jpg'
        shutil.move(old_path, new_path)
    
    else:
        new_path=r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Test\Dog\\'
        old_path=r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Dog\\'+str(i)+'.jpg'
        shutil.move(old_path, new_path)




#%% Okay lets actually build this model


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense 
from keras.preprocessing.image import ImageDataGenerator

#Okay, before we generate a CNN, we've got to set our hyperparameters
Filter_Size=3
Number_Filters=32
Input_Size=32
Maxpool_Size=2
Batch_Size=12
Steps=10000//Batch_Size
Epochs=10

model=Sequential()

#First add the Convolution layer
model.add(Conv2D(Number_Filters, (Filter_Size, Filter_Size),
                 input_shape=(Input_Size, Input_Size,3),
                 activation='relu'))

#Next add the max pooling layer
model.add(MaxPooling2D(pool_size=(Maxpool_Size, Maxpool_Size)))

#We want two layers of convolution, so do it again
#First add the Convolution layer
model.add(Conv2D(Number_Filters, (Filter_Size, Filter_Size),
                 input_shape=(Input_Size, Input_Size,3),
                 activation='relu'))

#Next add the max pooling layer
model.add(MaxPooling2D(pool_size=(Maxpool_Size, Maxpool_Size)))

#Now, we want to compress the convolution layer output down to two dimensation
model.add(Flatten())

#Then add our dense layers

model.add(Dense(units=128, activation='relu'))

#Output is binary 'cat/dog'
model.add(Dense(units=1, activation='sigmoid'))

#Dropout layer
#model.add(Dropout(.50))

#Annnndd....compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Now the hard part -- reading all this shit in. We'll use the imkage data generator function 
train_dir=r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Train'
test_dir=r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Test'

training_data_generator = ImageDataGenerator(rescale=1/255)
training_set=training_data_generator.flow_from_directory(train_dir,
                                                         target_size=(Input_Size, Input_Size),
                                                         batch_size=Batch_Size,
                                                         class_mode='binary')

model.fit_generator(training_set, steps_per_epoch=Steps, epochs=Epochs,verbose=1)



#%% DEBUG -- Look for broken images
import cv2
import os
import time

bad_list=[]

#If file is missing '.jpg' raise a flag
new_path=r'C:\Users\mahon\Documents\Python Scripts\NN Practice\PetImages\Test\Dog\\'
files = os.listdir(new_path)
upper=int(len(files))

for file in files:
    extension=file[-4:]
    
    if extension != '.jpg':
        print('Error: invalid extension in' + file)

    #Try to open it
    else:
        try:
            time.sleep(.05) #Let it wait a second so the image reader doesn't get ahead of itself
            img=cv2.imread(new_path+file)
            size=img.shape
        except:
            print(f'file {file} is not a valid image file ')
            bad_list.append(file)

####Image 666 for cats and 11702 for dogs are invalid



