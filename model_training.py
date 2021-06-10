from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from sklearn import preprocessing

import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import numpy as np

import random

import cv2 as cv
import os
import seaborn as sns


#fashion_mnist = keras.datasets.fashion_mnist

# example of progressively loading images from file
from keras.preprocessing.image import ImageDataGenerator
# create generator

categories = ['black widow', 'captain america', 'doctor strange', 'hulk', 'ironman',
               'loki', 'spider-man', 'thanos']

batch_size = 32

img_width, img_height = 64, 64

trainDir= "C:\\Users\\N0731739\\Downloads\\marvel\\train"
testDir = "C:\\Users\\N0731739\\Downloads\\marvel\\test"

nb_train_samples = 2584
nb_validation_samples = 451



##############make train data
train_data=[]

for category in categories:
    
        #each cateogry into unique integer
        label=categories.index(category)
        path=os.path.join(trainDir,category)
        
        for img_file in os.listdir(path):
            
            img=cv.imread(os.path.join(path,img_file),1)
            img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
            img=cv.resize(img,(60,60))            
            train_data.append([img,label])



##############make test data
test_data=[]

for category in categories:
       
        #each cateogry into unique integer
        label=categories.index(category)
        path=os.path.join(testDir,category)
        
        for img_file in os.listdir(path):
            
            img=cv.imread(os.path.join(path,img_file),1)
            img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
            img=cv.resize(img,(60,60))
            test_data.append([img,label])
            
###############lets seprate the feature and target variable
train_X=[]
train_y=[]

for features,label in train_data:
    train_X.append(features)
    train_y.append(label)



################lets seprate the feature and target variable
test_X=[]
test_y=[]

for features,label in test_data:
    test_X.append(features)
    test_y.append(label)



#convert image array to numpy array
#-1 means same size
# 40*40 means height and width
# 3 for R+G+B
train_X=np.array(train_X).reshape(-1,60,60,3)
train_X=train_X/255.0
train_X.shape

test_X=np.array(test_X).reshape(-1,60,60,3)
test_X=test_X/255.0
test_X.shape

#we divide the np array by 255 to close all values to 0


#convert label into the one hot encode
from keras.utils import to_categorical
#train y
one_hot_train=to_categorical(train_y)

#test_y
one_hot_test=to_categorical(test_y)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout,Activation

    
model = Sequential()

# CNN model: currently not overfitted (train acc = 80%, test acc = 82%)
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(60,60,3)))
model.add(MaxPooling2D(pool_size=(2,2))) # groups pixels in image and filters them into subset: input is reduced but features remain the same (compresses and extracts features
model.add(Dropout(0.20))

model.add(Conv2D(64, (3, 3),  activation="relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.20))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.40))

model.add(Conv2D(256, (3, 3),  activation="relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.20))

model.add(Flatten())

model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(256,activation='relu'))
          
model.add(Dense(8,activation='softmax'))




#print(model.summary())

##model.compile(optimizer="rmsprop",
##              loss="binary_crossentropy", 
##              metrics=["accuracy"])

#we will choose adam optimizer
#we have 4 categories so loss function is categorical_crossentropy
#metrics accuracy
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    




         # fit traininig data to model

         #lets split the 20% train dataset for validation 
hist=model.fit(train_X,one_hot_train,epochs=50,batch_size=32,validation_split=0.2)

##model.fit(
##    train_gen,
##    steps_per_epoch = nb_train_samples,
##    epochs = 25,
##    vadlidation_data = test_gen,
##    validation_steps = nb_validation_samples // batch_size)


test_loss,test_acc=model.evaluate(test_X,one_hot_test)
test_loss,test_acc

model.save("marvel-cnn.h5")

#train and validation loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train','Validation'],loc='upper left')
plt.show()

#train and validation accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train','Validation'],loc='upper left')
plt.show()

y_pred=model.predict_classes(test_X)
print(y_pred)

for i in range(10):
	print("Actual=%s, Predicted=%s" % (test_y[i], y_pred[i]))

from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(test_y,y_pred)



#sns.heatmap(confusion_matrix(test_y,y_pred))

#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#print('Test accuracy:', test_acc) # prints accuracy accuracy on test dataset


#predictions = model.predict(test_images)

#
#def plot_image(i, predictions_array, true_label, img):
#  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
#  plt.grid(False)
#  plt.xticks([])
#  plt.yticks([])
#  
#  plt.imshow(img, cmap=plt.cm.binary)
#  
#  predicted_label = np.argmax(predictions_array)
#  if predicted_label == true_label:
#    color = 'blue'
#  else:
#    color = 'red'
#  
#  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                100*np.max(predictions_array),
#                                class_names[true_label]),
#                                color=color)
#    
#        
#
#def plot_value_array(i, predictions_array, true_label):
#  predictions_array, true_label = predictions_array, true_label[i]
#  plt.grid(False)
#  plt.xticks([])
#  plt.yticks([])
#  thisplot = plt.bar(range(10), predictions_array, color="#777777")
#  plt.ylim([0, 1])
#  predicted_label = np.argmax(predictions_array)
#  
#  thisplot[predicted_label].set_color('red')
#  thisplot[true_label].set_color('blue')
#    

#i = random.randint(1, 10000)
#plt.figure(figsize=(6,3))
#plt.subplot(1,2,1)
#plot_image(i, predictions[i], test_labels, test_images)
#plt.subplot(1,2,2)
#plot_value_array(i, predictions[i],  test_labels)
#plt.show()


#img = test_images[1]
#print(img.shape)
#
#img = (np.expand_dims(img,0))
#print(img.shape)

#predictions_single = model.predict(img)
#print(predictions_single)

#plot_value_array(1, predictions_single[0], test_labels)
#plt.xticks(range(10), class_names, rotation=45)
#plt.show()

#prediction_result = np.argmax(predictions_single[0])
#print(prediction_result) #predicts label 
