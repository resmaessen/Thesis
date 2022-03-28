# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:40:19 2022

@author: Rosa Maessen
"""

import os
from params import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from human_robot_weights import human_robot
from robot_robot_weights import robot_robot
import time
from pDMP_functions import pDMP
from scipy.fft import fft, fftfreq, rfft, rfftfreq, ifft, irfft



plt.close('all')

runs = 10

fig1, ax1 = plt.subplots(1,1, figsize=(20.0, 10.0), sharex=True)
#fig2, ax2 = plt.subplots(1,1, figsize=(20.0, 10.0), sharex=True)

for run in range(1,runs+1):
    w_traject = np.loadtxt('save_data/data_w_traject_robot_'+str(run)+'.csv', unpack=True)
    w_stiff = np.loadtxt('save_data/data_w_stiff_robot_'+str(run)+'.csv', unpack=True)
    
    DMP_traject = pDMP(DOF, N, alpha, beta, lambd, dt, h)
    DMP_stiff = pDMP(DOF, N, alpha, beta, lambd, dt, h)
    
    if DOF == 1:
        DMP_traject.set_weights(0, w_traject)
        DMP_stiff.set_weights(0, w_stiff)
    else:
        for i in range(DOF):
            DMP_traject.set_weights(i, w_traject[i,:])
            DMP_stiff.set_weights(i, w_stiff[i,:])
    
    
    DMP_traject.set_state(np.array([L]))
    
    
    phase = [2*np.pi*(dt/tau)*i for i in range(samples)] 
    
    data = []
    
    for idx in range(int(samples)):
        # generate phase
        phi = phase[idx]
        
        # Get trajectory      
        DMP_traject.set_phase( np.array([phi]) )
        DMP_traject.set_period( np.array([tau]) )
        
        DMP_stiff.set_phase( np.array([phi]) )
        DMP_stiff.set_period( np.array([tau]) )
        
        DMP_traject.repeat()
        DMP_stiff.repeat()
        
        DMP_traject.integration()
        DMP_stiff.integration()
        
        x, _, _ , _ = DMP_traject.get_state()
        k, _, _, _ = DMP_stiff.get_state()
        
        data.append([x[0], k[0]])
        
    data = np.asarray(data)
    
    
    yf = rfft(data[:,0])
    xf = rfftfreq(samples, dt)
    yr = irfft(yf)
    
    ax1.plot(xf, np.abs(yf), label = 'run '+str(run))
ax1.legend()