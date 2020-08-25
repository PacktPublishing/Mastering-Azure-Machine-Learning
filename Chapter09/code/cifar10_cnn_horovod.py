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
from keras_azure_ml_cb import AzureMlKerasCallback

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
parser.add_argument('--epochs', type=int, dest='epochs', default=10, help='epochs')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.0001, help='learning rate')
parser.add_argument('--decay', type=float, dest='decay', default=1e-6, help='decay')
args = parser.parse_args()

import horovod.keras as hvd
hvd.init()

from keras import backend as K
import tensorflow as tf

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

batch_size = args.batch_size
num_classes = 10
epochs = args.epochs
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

# Horovod: adjust learning rate based on number of GPUs.
opt = keras.optimizers.rmsprop(lr=args.learning_rate * hvd.size(), decay=args.decay)

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

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

# Create Horovod initialization callback
callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0)
]

if hvd.rank() == 0:
    callbacks += [azureml_cb, checkpoint_cb, logger_cb]

model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        verbose=1 if hvd.rank() == 0 else 0,
        callbacks=callbacks)

# Load the best model
if hvd.rank() == 0:
    model = load_model(model_path)

    # Score trained model
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    run.log('Test loss', scores[0])
    print('Test accuracy:', scores[1])
    run.log('Test accuracy', scores[1])
