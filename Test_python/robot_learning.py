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

def train_robot(data_robot, data_human, DOF, N, alpha,beta,lambd, dt,tau):

    data = []
    data_robot2 = []

    y_old = 0
    dy_old = 0
    myDMP = pDMP(DOF, N, alpha, beta, lambd, dt)
    phi = 0 # DMP phase

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
        data.append([time,phi, x[1], x[1]-data_robot[i,1], x[1]-data_human[i]])
        data_robot2.append(x.tolist())
        
        
    data = np.asarray(data)
    data_robot2 = np.asarray(data_robot2)
    data_robot2 = np.concatenate((data_robot2[int(0.5*samples):, :],data_robot2[int(0.5*samples):, :]))
    
    
    return data, data_robot2


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
data_human = []
data_robot = []

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
    data.append([time,phi,x[1], x[1]-y[1], x[1]-y[1]])
    data_human.append(y.tolist())
    data_robot.append(x.tolist())


data1 = np.asarray(data)
data_human = np.asarray(data_human)
data_robot1 = np.asarray(data_robot)

data_robot1 = np.concatenate((data_robot1[int(0.5*samples):, :],data_robot1[int(0.5*samples):, :]))

trials = 20

data_big =  np.empty([trials+1,data1.shape[0],data1.shape[1]])
data_big_robot =  np.empty([trials+1,data_robot1.shape[0],data_robot1.shape[1]])

data_big[0,:,:] = data1
data_big_robot[0,:,:] = data_robot1

for i in range(trials):
    data_big[i+1,:,:], data_big_robot[i+1,:,:] = train_robot(data_big_robot[i,:,:], data_human[:,1], DOF, N, alpha,beta,lambd, dt,tau)


''' Plot the figure '''

fig = plt.figure()
# set height ratios for subplots
gs = gridspec.GridSpec(3, 1, height_ratios = [3,1,1]) 

# the first subplot
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)
ax2 = plt.subplot(gs[2], sharex = ax1)

example_traject, = ax0.plot(data1[:,0],data_human[:,1],'r', label = 'ex')

for i in range(len(data_big[:,0,0])):
    if i%4 == 0:
       ax0.plot(data_big[i,:,0],data_big[i,:,2], label = str(i))
       ax1.plot(data_big[i,:,0], data_big[i,:,3], label = str(i))
       ax2.plot(data_big[i,:,0], data_big[i,:,4], label = str(i))
        


ax0.axvline(x = data_big[0,int(samples/2),0], color="grey")
ax1.axvline(x = data_big[0,int(samples/2),0], color="grey")
ax2.axvline(x = data_big[0,int(samples/2),0], color="grey")

plt.setp(ax0.get_xticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)

# remove last tick label for the second subplot
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
yticks2 = ax2.yaxis.get_major_ticks()
yticks2[-1].label1.set_visible(False)

ax0.set_xlim([0,max(data_big[0,:,0])])
ax1.set_xlim([0,max(data_big[0,:,0])])

# put legend on first subplot
ax0.legend( loc='lower right')


ax0.set_ylabel('signal', fontsize='13')
ax1.set_ylabel('error w.r.t. expert', fontsize='13')
ax2.set_ylabel('error w.r.t. human', fontsize='13')
ax2.set_xlabel('time [s]', fontsize='12')

# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)
plt.show()




