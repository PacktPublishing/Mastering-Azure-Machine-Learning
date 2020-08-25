'''
#Train a simple deep CNN on the CIFAR10 small images dataset.

Source: https://raw.githubusercontent.com/keras-team/keras/master/examples/cifar10_cnn.py
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger
from azureml.core import Run
import os
import logging
import sys
from keras_azure_ml_cb import AzureMlKerasCallback

batch_size = 32
num_classes = 10
epochs = 10
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'outputs')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Save model and weights with checkpoint callback
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
checkpoint_cb = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

# Log output of the script
if not os.path.isdir('logs'):
    os.makedirs('logs')

# Logging with debug level
logging.basicConfig(filename='logs/debug.log', filemode='w', level=logging.DEBUG)

# Log training iterations to file
logger_cb = CSVLogger('logs/training.log')

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Load the current run
run = Run.get_context()

# Create an Azure Machine Learning monitor callback
azureml_cb = AzureMlKerasCallback(run)

model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        verbose=0,
        callbacks=[azureml_cb, checkpoint_cb, logger_cb])

# Load the best model
model = load_model(model_path)

# Score trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
run.log('Test loss', scores[0])
print('Test accuracy:', scores[1])
run.log('Test accuracy', scores[1])
