# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:04:38 2022

@author: Rosa Maessen
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pDMP_functions import pDMP
from matplotlib import gridspec 
from IPython.display import HTML

plt.close('all')

anim = 0

# EXPERIMENT PARAMETERS
dt = 0.05 # system sample time
exp_time = 200 # total experiment time
samples = int(1/dt) * exp_time

tau = 5 # Time period = 1/frequency (NOTE: this is the frequency of a period)
phi = 0 # Starting phase

L = 0.5 # Length saw

K1max = 100 # N/m
K2max = 100 # N/m

c1 = 50
c2 = 50

m = 10 # kg

K1 = []
K2 = []
time_ = []
phase = []

a = 0.5

phase = [a * 2*np.pi*(dt/tau)*i for i in range(samples)]



for i in range(samples):
    if np.sin(phase[i])>=0:
        K2.append(K2max)
    else: 
        K2.append(0)
    

K2 = np.asarray(K2)


DOF = 1 # degrees of freedom (number of DMPs to be learned)
N = 25 # number of weights per DMP (more weights can reproduce more complicated shapes)
alpha = 8 # DMP gain alpha
beta = 2 # DMP gain beta
lambd = 0.995 # forgetting factor
tau = 5 # DMP time period = 1/frequency (NOTE: this is the frequency of a periodic DMP)
phi = 0 # DMP phase

mode = 1 # DMP mode of operation (see below for details)

y_old = K2[0]
dy_old = 0

data = []

# create a DMP object
DMP_traject = pDMP(DOF, N, alpha, beta, lambd, dt)


# Human to Robot

''' Phase 1'''
for i in range ( int(samples*2/3 )):

    # generate phase
    phi = phase[i]
    
   
                  
    
    # generate an example trajectory (e.g., the movement that is to be learned)
    y = np.array([K2[i]])
    # calculate time derivatives
    dy = (y - y_old) / dt 
    ddy = (dy - dy_old) / dt
    
    # generate an example update (e.g., EMG singals that update exoskeleton joint torques as in [Peternel, 2016])
    U = 10*y # typically update factor is an input signal multiplied by a gain
    
    # set phase and period for DMPs
    DMP_traject.set_phase( np.array([phi]) )
    DMP_traject.set_period( np.array([tau]) )
    

    
    # DMP mode of operation
    if i < int( 1/3 * samples ): # learn/update for half of the experiment time, then repeat that DMP until the end
        if( mode == 1 ):
            DMP_traject.learn(y, dy, ddy) # learn DMP based on a trajectory
        elif ( mode == 2 ):
            DMP_traject.update(U) # update DMP based on an update factor
    else:
        DMP_traject.repeat() # repeat the learned DMP
    
    # DMP integration
    DMP_traject.integration()
    
    
    # old values	
    y_old = y
    dy_old = dy
    
    # store data for plotting
    x, dx, ph, ta = DMP_traject.get_state()
    time = dt*i
    data.append([time,phi,x[0], y[0], x[0]-y[0]])


data = np.asarray(data)



plt.figure()
plt.plot(data[:,0], data[:,2])
plt.plot(data[:,0], data[:,3])
plt.show()
        
