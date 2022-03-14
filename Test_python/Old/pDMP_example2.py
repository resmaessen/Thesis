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


plt.close("all")


# EXPERIMENT PARAMETERS
dt = 0.1 # system sample time
exp_time = 80 # total experiment time
samples = int(1/dt) * exp_time

DOF = 4 # degrees of freedom (number of DMPs to be learned)
N = 25 # number of weights per DMP (more weights can reproduce more complicated shapes)
alpha = 8 # DMP gain alpha
beta = 2 # DMP gain beta
lambd = 0.995 # forgetting factor
tau = 5 # DMP time period = 1/frequency (NOTE: this is the frequency of a periodic DMP)
phi = 0 # DMP phase

mode = 1 # DMP mode of operation (see below for details)

y_old = 0
dy_old = 0

data = []
data_robot = []
data_human = []
# create a DMP object
myDMP = pDMP(DOF, N, alpha, beta, lambd, dt)


# Coefficients
a, b = 1, 0

# Human to Robot
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
    time = dt*i
    data.append([time,phi,x[1],y[1], x[1]-y[1]])
    data_robot.append(x.tolist())
    data_human.append(y.tolist())


data = np.asarray(data)
data_robot = np.asarray(data_robot)
data_human = np.asarray(data_human)

data2 = []

y_old = 0
dy_old = 0
myDMP2 = pDMP(DOF, N, alpha, beta, lambd, dt)

# Robot to Robot
for i in range ( samples ):

    # generate phase
    phi += 2*np.pi * dt/tau
    
    # generate an example trajectory (e.g., the movement that is to be learned)
    y = data_robot[i]
    # calculate time derivatives
    dy = (y - y_old) / dt 
    ddy = (dy - dy_old) / dt
    
   # phi = np.arcsin(y[0])
    
    # generate an example update (e.g., EMG singals that update exoskeleton joint torques as in [Peternel, 2016])
    U = 10*y # typically update factor is an input signal multiplied by a gain
    
    
    # set phase and period for DMPs
    myDMP2.set_phase( np.array([phi,phi,phi,phi]) )
    myDMP2.set_period( np.array([tau,tau,tau,tau]) )
    
    
    
    # DMP mode of operation
    if i < int( 0.5 * samples ): # learn/update for half of the experiment time, then repeat that DMP until the end
        if( mode == 1 ):
            myDMP2.learn(y, dy, ddy) # learn DMP based on a trajectory
        elif ( mode == 2 ):
            myDMP2.update(U) # update DMP based on an update factor
    else:
        myDMP2.repeat() # repeat the learned DMP
    # DMP integration
    myDMP2.integration()
    
    
    # old values	
    y_old = y
    dy_old = dy
    
    # store data for plotting
    x, dx, ph, ta = myDMP2.get_state()
    time = dt*i
    data2.append([time,phi, x[1], x[1]-data_robot[i,1]], x[1]-data_human[i,1])

data2 = np.asarray(data2)
# PLOTS

plt.close("all")
fig = plt.figure()
# set height ratios for subplots
gs = gridspec.GridSpec(3, 1, height_ratios = [3,1,1]) 

#gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1]) 

# the first subplot
ax0 = plt.subplot(gs[0])
example_traject, = ax0.plot(data[:,0],data[:,3],'r')
dmp_traject, = ax0.plot(data[:,0],data[:,2],'b')
dmp_traject2, = ax0.plot(data2[:,0],data2[:,2],'g')
ax0.axvline(x = data[int(samples/2),0], color="grey")

# the second subplot
# shared axis X
ax1 = plt.subplot(gs[1], sharex = ax0)
error_, = ax1.plot(data[:,0], data[:,4], color='b')
error2_, = ax1.plot(data2[:,0], data2[:,3], color='g')
ax1.axvline(x = data[int(samples/2),0], color="grey")

ax2 = plt.subplot(gs[2], sharex = ax1)
error2_2, = ax2.plot(data[:,0], data2[:,3], color='g')
ax2.axvline(x = data[int(samples/2),0], color="grey")


plt.setp(ax0.get_xticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)

# remove last tick label for the second subplot
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
yticks2 = ax2.yaxis.get_major_ticks()
yticks2[-1].label1.set_visible(False)

ax0.set_xlim([0,max(data[:,0])])
ax1.set_xlim([0,max(data[:,0])])

# put legend on first subplot
ax0.legend((example_traject, dmp_traject, dmp_traject2), ('example', 'dmp','dmp2'), loc='lower right')

ax0.set_ylabel('signal', fontsize='13')
ax1.set_ylabel('error', fontsize='13')
ax1.set_xlabel('time [s]', fontsize='12')

# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)
plt.show()