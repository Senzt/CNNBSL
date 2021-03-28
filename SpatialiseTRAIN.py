# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 18:43:20 2021

@author: Waradon Senzt Phokhinanan
"""

############################################################################################

import math
import librosa
import scipy.io
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import os
import soundfile as sf

############################################################################################

def AdjustNoiseBySNR(signal,noise,SNR):
    RMS_s = math.sqrt(np.mean(signal**2))
    RMS_n = math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    RMS_nd = math.sqrt(np.mean(noise**2))
    noise = noise*(RMS_n/RMS_nd) 
    return noise

############################################################################################
    
def NoiseTrainingImport():
    
    NoiseData = []
    NoiseList = os.listdir('./NoiseTRAIN')    
    for NoiseX in NoiseList:
        NoisePATH = './NoiseTRAIN/' + NoiseX
        (NoiseI, rate) = librosa.load(NoisePATH, sr=16000)
        NoiseI /= np.max(np.abs(NoiseI),axis=0)
        NoiseData.append(NoiseI)
        
    return NoiseData
        
############################################################################################

def spatialise37azimuths(speech_name, SNR, NoiseSig, NoiseNum):
    
    #Import speech
    Selected_speech = '.' + speech_name
    (sig, rate) = librosa.load(Selected_speech, sr=16000)
    sig /= np.max(np.abs(sig),axis=0)

    #Import MIT_HRIR
    MITmat = scipy.io.loadmat('hrir_MIT.mat')
    HRIRMIT = MITmat['hrir_MIT']

    #This model only focuses on azimuth prediction, so elevation at 0 is selected.
    HRIR_L0ele = HRIRMIT[:,:,0]
    HRIR_R0ele = HRIRMIT[:,:,1]
    
    #Reduce HRIRs sampling to the same rate as speech
    #Actually, I feel using the original sampling is better.
    HRIR_L0ele = librosa.resample(HRIR_L0ele, 44100, 16000)  
    HRIR_R0ele = librosa.resample(HRIR_R0ele, 44100, 16000)

    #Add diffuse noise
    #This model simply adds the diffuse noises by using the noise in front of the speaker (equal energy on left/right).
    #Therefore, apply the noise at 0 azimuth to the other 36 azimuths.
    
    #Also, roughly both left and right channels will be added with the same position/length of an input noise file. 
    #You can adjust to train with random position of the input noise file.
    
    #The correct way to add diffuse noise should sum over the noises on the left and right for each target azimuth, 
    #or by using real environment recording of noise database.
    HRIRLeft0 = HRIR_L0ele[19,:]
    HRIRRight0 = HRIR_R0ele[19,:]

    NewSigLEFT0 = signal.convolve(sig, HRIRLeft0)
    NewSigRIGHT0 = signal.convolve(sig, HRIRRight0)

    NoiseAdjustSNRLEFT=AdjustNoiseBySNR(NewSigLEFT0,NoiseSig,SNR)
    NoiseAdjustSNRRIGHT=AdjustNoiseBySNR(NewSigRIGHT0,NoiseSig,SNR)

    ################ Spatialise 37 azimuths from -90 to 90 ################

    ILDIPD_Feature = []
    ILDIPD_Label = []

    for x in range(0, 37):
      HRIRLeftEx = HRIR_L0ele[x,:]
      HRIRRightEx = HRIR_R0ele[x,:]

      NewSigLEFT = signal.convolve(sig, HRIRLeftEx)
      NewSigRIGHT = signal.convolve(sig, HRIRRightEx)
      
      ##If the diffuse noise is too short, you can further manage the noise here.
      ##As I attempted only to predict the azimuth with 10–50 frames of feature, and my noise database has long duration, I have not managed this. 
      # if(len(NoiseAdjustSNRLEFT)<len(NewSigLEFT)):
      #     print('Diffuse noise is too short, please change the noise')
      # if(len(NoiseAdjustSNRRIGHT)>len(NewSigRIGHT)):
      #     print('Diffuse noise is too short, please change the noise')
          
      NewSigLEFT = NewSigLEFT+NoiseAdjustSNRLEFT[0:len(NewSigLEFT)]         
      NewSigRIGHT = NewSigRIGHT+NoiseAdjustSNRRIGHT[0:len(NewSigRIGHT)]

      # #You can uncomment this section to extract the sptialised speech
      # #test extraction
      # NewSTERIO = np.array([NewSigLEFT,NewSigRIGHT])
      # NewSTERIO = np.transpose(NewSTERIO)
      # speech_nameC = speech_name.replace('/', '_')
      # Exportname = 'Export-BinSP' + str(x) + '_' + str(NoiseNum) + '_' + str(speech_nameC)
      # sf.write(Exportname, NewSTERIO, 16000, 'PCM_24')

      ################ Feature Extraction ################

      #Do STFT by using Librosa library
      #• The window length is 40ms (640 samples)
      #• The hop size is 20ms (320 samples)
      #• So, the frequency bin is 321 (from 0 to 320), which represent from 0 to 8000 Hz
      #• The window type is Hamming
      STFTLeft = librosa.stft(NewSigLEFT, n_fft=640, hop_length=320, win_length=640, window='hamm', center=False, dtype=None, pad_mode='reflect')
      STFTRight = librosa.stft(NewSigRIGHT, n_fft=640, hop_length=320, win_length=640, window='hamm', center=False, dtype=None, pad_mode='reflect')

      IPD = np.angle(STFTRight/STFTLeft)
      ILD = 20*np.log10(np.abs(STFTRight)/np.abs(STFTLeft))
      IPDILD = np.array([IPD[:,20:70], ILD[:,20:70]])
      IPDILD = np.moveaxis(IPDILD, 0, -1)

      ILDIPD_Feature.append(IPDILD)
      ILDIPD_Label.append(x)

    ILDIPD_Feature = np.array(ILDIPD_Feature)
    ILDIPD_Label = np.array(ILDIPD_Label)

    return ILDIPD_Feature, ILDIPD_Label

############################################################################################
# MAIN PROGRAMME ###########################################################################
############################################################################################
    
azimuthdict = {
        0: "-90",
        1: "-85",
        2: "-80",
        3: "-75",
        4: "-70",
        5: "-65",
        6: "-60",
        7: "-55",
        8: "-50",
        9: "-45",
        10: "-40",
        11: "-35",
        12: "-30",
        13: "-25",
        14: "-20",
        15: "-15",
        16: "-10",
        17: "-5",
        18: "0",
        19: "5",
        20: "10",
        21: "15",
        22: "20",
        23: "25",
        24: "30",
        25: "35",
        26: "40",
        27: "45",
        28: "50",
        29: "55",
        30: "60",
        31: "65",
        32: "70",
        33: "75",
        34: "80",
        35: "85",
        36: "90"
    }

TRAIN_ILDIPD_FeatureCON = np.empty([0,321,50,2])
TRAIN_ILDIPD_LabelCON = np.empty([0])

NoiseData = NoiseTrainingImport()

#Generate Training Data

SpeechTrainD = os.listdir('./SpeechTRAIN')
for FileXD in SpeechTrainD:

  FileX = '/SpeechTRAIN/' + FileXD

  for SNRx in [-10,-5,0,5,10,15,20,25,30]:

    for Nx in range(0,len(NoiseData)):
        print('Spatialising')
        print('SNR: ' + str(SNRx))
        print('Speech file: ' + str(FileXD))
        print('Noise number: ' + str(Nx))
        NoisePUT = NoiseData[Nx]
        ILDIPD_Feature, ILDIPD_Label = spatialise37azimuths(FileX,SNRx,NoisePUT,Nx)
        TRAIN_ILDIPD_FeatureCON = np.vstack([TRAIN_ILDIPD_FeatureCON,ILDIPD_Feature])
        TRAIN_ILDIPD_LabelCON = np.hstack([TRAIN_ILDIPD_LabelCON,ILDIPD_Label.astype(int)])

######

TRAIN_ILDIPD_LabelCON = np.vectorize(azimuthdict.get)(TRAIN_ILDIPD_LabelCON)
TRAIN_ILDIPD_LabelCON = TRAIN_ILDIPD_LabelCON.astype(int)

with open('BinSL_TRAINextract.npy', 'wb') as f:
    np.save(f, TRAIN_ILDIPD_FeatureCON)
    np.save(f, TRAIN_ILDIPD_LabelCON)

print('Genrating training data has done!')
print('Total training samples: ' + str(TRAIN_ILDIPD_FeatureCON.shape))
print('Total training labels: ' + str(TRAIN_ILDIPD_LabelCON.shape))