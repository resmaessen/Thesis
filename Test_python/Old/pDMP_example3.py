# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 09:25:12 2022

@author: Rosa Maessen
"""

"""
PERIODIC DYNAMIC MOVEMENT PRIMITIVES (pDMP)

An example of how to use pDMP functions.


AUTHOR: Luka Peternel
e-mail: l.peternel@tudelft.nl


REFERENCE:
L. Peternel, T. Noda, T. Petrič, A. Ude, J. Morimoto and J. Babič
Adaptive control of exoskeleton robots for periodic assistive behaviours based on EMG feedback minimisation,
PLOS One 11(2): e0148942, Feb 2016

"""

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import gridspec 
from pDMP_functions import pDMP
import time


plt.close("all")


# EXPERIMENT PARAMETERS
dt = 0.1 # system sample time
exp_time = 80 # total experiment time


DOF = 4 # degrees of freedom (number of DMPs to be learned)
N = 25 # number of weights per DMP (more weights can reproduce more complicated shapes)
alpha = 8 # DMP gain alpha
beta = 2 # DMP gain beta
lambd = 0.995 # forgetting factor
tau = 5 # DMP time period = 1/frequency (NOTE: this is the frequency of a periodic DMP)
phi = 0 # DMP phase

mode = 1 # DMP mode of operation (see below for details)

# Coefficients
a, b = 1, 0

dt_ = [0.01, 0.1, 0.5]

c1 = ['blue', 'red', 'green']
tt = []


fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)

for idx, dt in enumerate(dt_):
    
    
    start = time.time()
    y_old = 0
    dy_old = 0
    time_ = 0
    
    samples = int(1/dt) * exp_time
    
    data = []
    
    # create a DMP object
    myDMP = pDMP(DOF, N, alpha, beta, lambd, dt)
    
    
    
    # MAIN LOOP
    for i in range ( samples ):
    
        # generate phase
        phi += 2*np.pi * dt/tau
        
        
        # generate an example trajectory (e.g., the movement that is to be learned)
        y = np.array([np.sin(a*phi)+np.cos(b*phi), np.cos(a*phi)+np.sin(b*phi), -np.sin(a*phi)-np.cos(b*phi), -np.cos(a*phi)-np.sin(b*phi)])
        # calculate time derivatives
        dy = (y - y_old) / dt 
        ddy = (dy - dy_old) / dt
        
        # generate an example update (e.g., EMG singals that update exoskeleton joint torques as in [Peternel, 2016])
        U = 10*y # typically update factor is an input signal multiplied by a gain
        
        # set phase and period for DMPs
        myDMP.set_phase( np.array([phi,phi,phi,phi]) )
        myDMP.set_period( np.array([tau,tau,tau,tau]) )
        
        
        
        # DMP mode of operation
        if i < int( 0.5 * samples ): # learn/update for half of the experiment time, then repeat that DMP until the end
            if( mode == 1 ):
                myDMP.learn(y, dy, ddy) # learn DMP based on a trajectory
            elif ( mode == 2 ):
                myDMP.update(U) # update DMP based on an update factor
        else:
            myDMP.repeat() # repeat the learned DMP
        
        # DMP integration
        myDMP.integration()
        
        
        # old values	
        y_old = y
        dy_old = dy
        
        # store data for plotting
        x, dx, ph, ta = myDMP.get_state()
        time_ = dt*i
        data.append([time_,phi,x[1],y[1], x[1]-y[1]])
    
    data = np.asarray(data)
    
    end = time.time()
    
    tt.append(end-start)
    
    dmp_traject, = ax0.plot(data[:,0],data[:,2], color = c1[idx], label = str(dt))
    error_, = ax1.plot(data[:,0], data[:,4], color=c1[idx])

# PLOTS
example_traject, = ax0.plot(data[:,0],2*data[:,3], color = 'yellow', linestyle = '--')



# the second subplot
# shared axis X


plt.setp(ax0.get_xticklabels(), visible=False)

# remove last tick label for the second subplot
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

# put legend on first subplot
#ax0.legend((example_traject, dmp_traject), ('example', 'dmp'), loc='lower left')
ax0.legend(loc = 'lower left')

ax0.set_ylabel('signal', fontsize='13')
ax1.set_ylabel('error', fontsize='13')
ax1.set_xlabel('time [s]', fontsize='12')

# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)
plt.show()
print(tt)

