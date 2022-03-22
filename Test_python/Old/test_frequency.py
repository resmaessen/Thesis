# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 09:35:57 2022

@author: Rosa Maessen
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pDMP_functions import pDMP
from matplotlib import gridspec 
from AFS_structure import AFS
from IPython.display import HTML
import random
from params import * 

plt.close('all')


t = np.linspace(0,5,samples)
y = np.zeros(len(t))
w = []
for idx, t_ in enumerate(t):
    
    w_ = np.pi*2
    y[idx] = np.sin(w_*t_) + np.cos(w_*3*t_)
    w.append(w_)


frequency_last = np.pi

myAFS = AFS(DOF, M, ni, K)
myAFS.set_flag(1)
myAFS.set_initial_AFS_state(frequency_last)

myAFS.set_initial_AFS_state(frequency_last)

data = []
phase_last = 0

dt = t[1]-t[0]

for i in range(len(t)):
        
   
    myAFS.update_input(0, np.array([y[i]]))

    myAFS.AFS_integrate(dt, 1)        
    
    output = myAFS.get_output(0)
    phase_last = myAFS.get_phase(0) 
    frequency_last = myAFS.get_frequency(0)
    
    
    data.append([output, phase_last, frequency_last])
data = np.asarray(data)

plt.figure()
plt.plot(t, y)

plt.figure()
plt.plot(t,w)
plt.plot(t,data[:,2])


    
    

