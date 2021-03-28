# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 20:09:36 2021

@author: Senzt
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout

def ModelCreate():
    model = modelLastEpoch = Sequential([
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
    
    return model

############################################################################################

with open('BinSL_TESTextract.npy', 'rb') as f:
    TEST_ILDIPD_FeatureCON = np.load(f)
    TEST_ILDIPD_LabelCON = np.load(f)
print('Loading testing data has done!')
print('Total samples: ' + str(TEST_ILDIPD_FeatureCON.shape))
print('Total labels: ' + str(TEST_ILDIPD_LabelCON.shape))

test_images = TEST_ILDIPD_FeatureCON
test_labels = TEST_ILDIPD_LabelCON

############################################################################################

modelLastEpoch = ModelCreate()

checkpoints_latest = 'checkpoints_every_epoch'
modelload = tf.train.latest_checkpoint(checkpoints_latest)
modelLastEpoch.load_weights(modelload)

predictionsLE = modelLastEpoch.predict(test_images)
predictionsLE = predictionsLE.flatten()
RangeAzimuthAcceptanceLE = abs(np.rint(predictionsLE)-test_labels)

print('Testing with the weights from last epoch')

prediction_accuracyLE10 = len(np.where(RangeAzimuthAcceptanceLE <= 10)[0]) / RangeAzimuthAcceptanceLE.shape[0] * 100
#print(len(np.where(RangeAzimuthAcceptanceLE <= 10)[0]))
print('prediction_accuracy +/-10 is ' + str(prediction_accuracyLE10))

prediction_accuracyLE5 = len(np.where(RangeAzimuthAcceptanceLE <= 5)[0]) / RangeAzimuthAcceptanceLE.shape[0] * 100
#print(len(np.where(RangeAzimuthAcceptanceLE <= 5)[0]))
print('prediction_accuracy +/- 5 is ' + str(prediction_accuracyLE5))

############################################################################################

# The best model, lowest RMSE on validation set
modelBestEpoch = ModelCreate()

checkpoints_best_only_path = 'checkpoints_best_only/checkpoint'    
modelBestEpoch.load_weights(checkpoints_best_only_path)

predictionsBE = modelBestEpoch.predict(test_images)
predictionsBE = predictionsBE.flatten()
RangeAzimuthAcceptanceBE = abs(np.rint(predictionsBE)-test_labels)

print('Testing with the weights from best epoch (lowest RME on validation)')

prediction_accuracyBE10 = len(np.where(RangeAzimuthAcceptanceBE <= 10)[0]) / RangeAzimuthAcceptanceBE.shape[0] * 100
#print(len(np.where(RangeAzimuthAcceptanceBE <= 10)[0]))
print('prediction_accuracy +/-10 is ' + str(prediction_accuracyBE10))

prediction_accuracyBE5 = len(np.where(RangeAzimuthAcceptanceBE <= 5)[0]) / RangeAzimuthAcceptanceBE.shape[0] * 100
#print(len(np.where(RangeAzimuthAcceptanceBE <= 5)[0]))
print('prediction_accuracy +/- 5 is ' + str(prediction_accuracyBE5))