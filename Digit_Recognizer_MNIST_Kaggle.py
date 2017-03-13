
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from keras.constraints import maxnorm

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

import cv2

nb_classes = 10
img_size = 28

# importing train data
data = pd.read_csv("/home/aiml_test_user/Shaheen/train_mnist.csv")

labels = data[[0]].values.ravel()
train = data.iloc[:,1:].values

train.shape

# convert to array, specify data type, and reshape
labels = labels.astype(np.uint8)
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)

train.shape

train1 = np.array(train).reshape((-1, 1, 28, 28)).astype('float32')

train1.shape

import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.imshow(train[125][0], cmap=cm.binary) # draw the picture


## dividing by 255
train1 /= 255


train1.shape


## splitting the data into train and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.3)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


## model
## Using 'sgd' and 'softmax'

def cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, img_size, img_size), activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    return model

model = cnn_model()


lr = 0.001
sgd = SGD(lr=lr, decay=1e-7, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])


def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))


## fitting the model
batch_size = 50
nb_epoch = 15

model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(x_test, Y_test))


# validation
validation = model.evaluate(x_test, Y_test, verbose=1)
print('Test accuracy:', validation[1])



## model 3
# Using 'adadelta' and 'sigmoid'

def cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, img_size, img_size), activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes, activation='sigmoid'))
    return model

model2 = cnn_model()

model2.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


## fitting the model
batch_size = 50
nb_epoch = 10

model2.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(x_test, Y_test))


#validation
validation = model2.evaluate(x_test, Y_test, verbose=1)
print('Test accuracy:', validation[1])


# importing test data
test = pd.read_csv("/home/aiml_test_user/Shaheen/test_mnist.csv")

print("type", type(train))
print('shape', train.shape)

## dividing by 255
test /= 255

test = np.array(test).reshape((-1, 1, 28, 28)).astype('float32')
test.shape


## predicting the test data
y_pred = model2.predict_classes(test)


# save results
np.savetxt('submission_digitRecognizer1.csv', np.c_[range(1,len(test)+1),y_pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')


# In[15]:

## data augmentation
X_train, X_val, Y_train, Y_val = train_test_split(train1, labels, test_size=0.2)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_val = np_utils.to_categorical(Y_val, nb_classes)

datagen = ImageDataGenerator(featurewise_center=False, 
                            featurewise_std_normalization=False, 
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10.,)

datagen.fit(X_train)


# In[16]:

# Reinstallise models 
## Using 'sgd' and 'softmax'


def cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, img_size, img_size), activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes, activation='softmax'))
    return model


model = cnn_model()

lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])


def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))


# In[24]:

# Reinstallise models 
# Using 'adadelta' and 'sigmoid'


def cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, img_size, img_size), activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes, activation='sigmoid'))
    return model


model1 = cnn_model()

model1.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])



# In[27]:

nb_epoch = 20
batch_size = 40
model1.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_val, Y_val))


# In[26]:

validation = model1.evaluate(X_val, Y_val, verbose=1)
print('Test accuracy:', validation[1])


# In[28]:

# predicting the test data
y_pred = model1.predict_classes(test)
# save results
np.savetxt('submission_digitRecognizer6.csv', np.c_[range(1,len(test)+1),y_pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')




