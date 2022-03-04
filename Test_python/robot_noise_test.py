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
from pDMP_functions import pDMP
from AFS_structure import AFS

plt.close('all')


# EXPERIMENT PARAMETERS
dt = 0.05 # system sample time
exp_time = 80 # total experiment time
samples = int(1/dt) * exp_time

DOF = 4 # degrees of freedom (number of DMPs to be learned)
N = 25 # number of weights per DMP (more weights can reproduce more complicated shapes)
N_AFS = 10
alpha = 8 # DMP gain alpha
beta = 2 # DMP gain beta
lambd = 0.995 # forgetting factor
tau = 5 # DMP time period = 1/frequency (NOTE: this is the frequency of a periodic DMP)
phi = 0 # DMP phase

ni = 2
K = 20


noise_max = 0.05
noise = np.random.normal(-noise_max,noise_max,[samples, DOF])

y = []
time_ = []
phi_ = []
for i in range(samples):
    phi += 2*np.pi * dt/tau
    phi_.append(phi)
    y_new = np.array([np.sin(phi), np.cos(phi), -np.sin(phi), -np.cos(phi)]) +noise[i,:]
    y.append(y_new.tolist())
    
    time_.append(dt*i)

data_robot = np.asarray(y)

myAFS = AFS(DOF, N_AFS, ni, K)
myAFS.set_flag(1)

freq_initial = 2
phi = [0, 0, 0, 0]

data_robot_b = []
phase_robot = []
frequency_init = np.ones(4)*freq_initial
frequency_last = frequency_init

myAFS.set_initial_AFS_state(frequency_init)

for i in range(samples):
   
    for j in range(DOF):
        myAFS.update_input(j, data_robot[i,j])
        myAFS.set_frequency(j,frequency_init[j])
        

    myAFS.AFS_integrate(dt)    
    data_robot_temp = []
    phase_temp = []
    
    for i in range(DOF):
        data_robot_temp.append(myAFS.get_output(i))
        phase_temp.append(myAFS.get_phase(i))
        frequency_last[i] = myAFS.get_frequency(i)
    
    phi = phase_temp
    
    data_robot_b.append([data_robot_temp[1], phi[1], frequency_last[1]])
    phase_robot.append(phi)
    
    

data_robot_b = np.asarray(data_robot_b)
phase_robot = np.asarray(phase_robot)



'''
phase_robot_ = np.mean(phase_robot)
for i in range(1,phase_robot.shape[0]):
    phase_robot[i] = phase_robot[i-1]+ phase_robot_

mode = 1 # DMP mode of operation (see below for details)

y_old = 0
dy_old = 0

data = []

# create a DMP object
myDMP = pDMP(DOF, N, alpha, beta, lambd, dt)

# MAIN LOOP
for i in range ( samples ):

    # generate phase
    phi = np.array([phi_[i], phi_[i], phi_[i], phi_[i]])
    
    # generate an example trajectory (e.g., the movement that is to be learned)
    y = data_robot[i,:]
    
    # calculate time derivatives
    dy = (y - y_old) / dt 
    ddy = (dy - dy_old) / dt
    
    # generate an example update (e.g., EMG singals that update exoskeleton joint torques as in [Peternel, 2016])
    U = 10*y # typically update factor is an input signal multiplied by a gain
    
    # set phase and period for DMPs
    myDMP.set_phase( phi )
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
    data.append([time,phi[1],x[1],y[1]])





plt.figure()
plt.plot(time_,data_robot[:,1],label = 'data')
plt.plot(time_,data_robot_b[:,0], 'r--', label = 'AFS')
plt.legend()
plt.show()


# PLOTS
data = np.asarray(data)

plt.figure()
# input
plt.plot(data[:,0],data[:,3],'r')
# DMP
plt.plot(data[:,0],data[:,2],'b')

plt.xlabel('time [s]', fontsize='12')
plt.ylabel('signal', fontsize='13')

plt.legend(['input','DMP'])

plt.title('Periodic DMP', fontsize='14')
plt.show()
'''

plt.figure()
plt.plot(time_, phase_robot[:,1])
plt.plot(time_, phi_)
plt.show()



