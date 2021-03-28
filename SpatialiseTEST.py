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
    
def NoiseTestingImport():
    
    NoiseData = []
    NoiseList = os.listdir('./NoiseTEST')    
    for NoiseX in NoiseList:
        NoisePATH = './NoiseTEST/' + NoiseX
        (NoiseI, rate) = librosa.load(NoisePATH, sr=16000)
        NoiseI /= np.max(np.abs(NoiseI),axis=0)
        NoiseData.append(NoiseI)
        
    return NoiseData
        
############################################################################################

def spatialise37azimuths(speech_name, SNR, NoiseSig, NoiseNum):
    
    Selected_speech = '.' + speech_name
    (sig, rate) = librosa.load(Selected_speech, sr=16000)
    sig /= np.max(np.abs(sig),axis=0)

    MITmat = scipy.io.loadmat('hrir_MIT.mat')
    HRIRMIT = MITmat['hrir_MIT']
    HRIR_L0ele = HRIRMIT[:,:,0]
    HRIR_R0ele = HRIRMIT[:,:,1]    
    HRIR_L0ele = librosa.resample(HRIR_L0ele, 44100, 16000)  
    HRIR_R0ele = librosa.resample(HRIR_R0ele, 44100, 16000)
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

      NewSigLEFT = NewSigLEFT+NoiseAdjustSNRLEFT[0:len(NewSigLEFT)]         
      NewSigRIGHT = NewSigRIGHT+NoiseAdjustSNRRIGHT[0:len(NewSigRIGHT)]

      # #You can uncomment this section to listen the sptialised speech
      # #test extraction
      # NewSTERIO = np.array([NewSigLEFT,NewSigRIGHT])
      # NewSTERIO = np.transpose(NewSTERIO)
      # speech_nameC = speech_name.replace('/', '_')
      # Exportname = 'Export-BinSP' + str(x) + '_' + str(NoiseNum) + '_' + str(speech_nameC)
      # sf.write(Exportname, NewSTERIO, 16000, 'PCM_24')

      ################ Feature Extraction ################

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

NoiseData = NoiseTestingImport()

TEST_ILDIPD_FeatureCON = np.empty([0,321,50,2])
TEST_ILDIPD_LabelCON = np.empty([0])

#Generate Testing Data

SpeechTestD = os.listdir('./SpeechTEST')
for FileXD in SpeechTestD:

  FileX = '/SpeechTEST/' + FileXD

  for SNRx in [-6,0,6]:

    for Nx in range(0,len(NoiseData)):
        print('Spatialising')
        print('SNR: ' + str(SNRx))
        print('Speech file: ' + str(FileXD))
        print('Noise number: ' + str(Nx))
        NoisePUT = NoiseData[Nx]
        ILDIPD_Feature, ILDIPD_Label = spatialise37azimuths(FileX,SNRx,NoisePUT,Nx)
        TEST_ILDIPD_FeatureCON = np.vstack([TEST_ILDIPD_FeatureCON,ILDIPD_Feature])
        TEST_ILDIPD_LabelCON = np.hstack([TEST_ILDIPD_LabelCON,ILDIPD_Label.astype(int)])

######

TEST_ILDIPD_LabelCON = np.vectorize(azimuthdict.get)(TEST_ILDIPD_LabelCON)
TEST_ILDIPD_LabelCON = TEST_ILDIPD_LabelCON.astype(int)

with open('BinSL_TESTextract.npy', 'wb') as f:
    np.save(f, TEST_ILDIPD_FeatureCON)
    np.save(f, TEST_ILDIPD_LabelCON)

print('Genrating testing data has done!')
print('Total testing samples: ' + str(TEST_ILDIPD_FeatureCON.shape))
print('Total testing labels: ' + str(TEST_ILDIPD_LabelCON.shape))