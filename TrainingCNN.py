# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 18:43:20 2021

@author: Waradon Senzt Phokhinanan
"""
  
############################################################################################

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint

############################################################################################

def get_checkpoint_best_only():    
    checkpoints_best_only_path = 'checkpoints_best_only/checkpoint'
    checkpoints_best_only = ModelCheckpoint(filepath=checkpoints_best_only_path,
                                            save_weights_only=True,
                                            save_freq='epoch',
                                            monitor='val_mean_squared_error',
                                            save_best_only=True,
                                            verbose=1)    
    return checkpoints_best_only

def get_checkpoint_every_epoch():    
    checkpoints_every_epoch_path = 'checkpoints_every_epoch/checkpoint_{epoch:03d}'
    checkpoints_every_epoch = ModelCheckpoint(filepath=checkpoints_every_epoch_path,
                                              frequency='epoch',
                                              save_weights_only=True,
                                              verbose=1)    
    return checkpoints_every_epoch

############################################################################################

#Import training data
with open('BinSL_TRAINextract.npy', 'rb') as f:
    TRAIN_ILDIPD_FeatureCON = np.load(f)
    TRAIN_ILDIPD_LabelCON = np.load(f)
    
print('Loading training data has done!')
print('Total samples: ' + str(TRAIN_ILDIPD_FeatureCON.shape))
print('Total labels: ' + str(TRAIN_ILDIPD_LabelCON.shape))

#Split into training (75%) and validation (25%) set by sklearn train_test_split()
train_images, x_v, train_labels, y_v = train_test_split(TRAIN_ILDIPD_FeatureCON,TRAIN_ILDIPD_LabelCON,test_size = 0.25,train_size =0.75)
print('Total training samples: ' + str(train_images.shape))
print('Total validation samples: ' + str(x_v.shape))
print('TensorFlow version: ' + tf.__version__)

############################################################################################

# Build the Sequential convolutional neural network model
model = Sequential([
    Conv2D(32, (5,5), kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=regularizers.l2(0.001), activation='relu',strides=3, input_shape=(321,50,2)),
    tf.keras.layers.BatchNormalization(),
    Conv2D(64, (3,3), kernel_regularizer=regularizers.l2(0.001), activation='relu',strides=2),
    tf.keras.layers.BatchNormalization(),
    Conv2D(96, (3,3), kernel_regularizer=regularizers.l2(0.001), activation='relu',strides=2),
    tf.keras.layers.BatchNormalization(),
    Conv2D(128, (3,3), kernel_regularizer=regularizers.l2(0.001), activation='relu',strides=2),
    tf.keras.layers.BatchNormalization(),
    Flatten(),
    Dense(1024,activation='relu'),
    Dropout(0.3),
    Dense(512,activation='relu'),
    Dense(256,activation='relu'),
    Dense(1),
])

model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
mse = tf.keras.metrics.MeanSquaredError()

model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=[mse]
             )

print(model.loss)
print(model.optimizer)
print(model.metrics)
print(model.optimizer.lr)

callbacks = [get_checkpoint_best_only(),get_checkpoint_every_epoch()]

# Fitting
history = model.fit(train_images, train_labels, epochs=500, validation_data=(x_v, y_v), batch_size=64, callbacks=callbacks)