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

plt.close('all')

anim = 0

# EXPERIMENT PARAMETERS
dt = 0.05 # system sample time
exp_time = 100 # total experiment time
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

a = 0.2

phase = [a * 2*np.pi*(dt/tau)*i for i in range(samples)]
K1_1 = np.asarray([K1max for i in range(samples)])
K1_2 = np.asarray([np.sin(phase[i]) * K1max for i in range(samples)])

xm_init = 0
x1_init = xm_init - L/2
x2_init = xm_init + L/2

xr1l = x1_init -L/2 # Goal position
xr1r = x1_init +L/2

x1_last = x1_init

dx1_last = 0 
dx2_last = dx1_last

data_1 = []
data_2 = []
data_1.append([x1_last, dx1_last])
data_2.append([x2_init, dx2_last])

time_.append(0)

for i in range(1,samples):
    
    
    if np.sin(phase[i])>=0:
        ddx = K1_1[i]/m *(xr1l - x1_last) - (c1/m)*dx1_last # +  K2[i]/m *(xr1r - x1_last) -  (c2/m)*dx1_last 
    else: 
        ddx = K1_1[i]/m * (xr1r-x1_last) -  (c1/m)*dx1_last 
        
    dx = ddx*dt + dx1_last
    x = dx*dt + x1_last
    
    data_1.append([x, dx])
    data_2.append([x + L, dx])
    
    x1_last = x
    dx1_last = dx
    time_.append(dt*i)
 
data_1 = np.asarray(data_1)
data_2 = np.asarray(data_2)

plt.figure()
plt.plot(time_, data_1)
plt.plot(time_, data_2)
plt.show()


DOF = 1 # degrees of freedom (number of DMPs to be learned)
N = 25 # number of weights per DMP (more weights can reproduce more complicated shapes)
alpha = 8 # DMP gain alpha
beta = 2 # DMP gain beta
lambd = 0.995 # forgetting factor
tau = 5 # DMP time period = 1/frequency (NOTE: this is the frequency of a periodic DMP)
phi = 0 # DMP phase

mode = 1 # DMP mode of operation (see below for details)

y_old = data_2[0,0]
dy_old = 0

data = []
data_robot = []

# create a DMP object
myDMP = pDMP(DOF, N, alpha, beta, lambd, dt)


# Coefficients
a, b = 1, 0

# Human to Robot
for i in range ( samples ):

    # generate phase
    phi = phase[i]
    
    # generate an example trajectory (e.g., the movement that is to be learned)
    y = np.array([data_2[i,0]])
    # calculate time derivatives
    dy = (y - y_old) / dt 
    ddy = (dy - dy_old) / dt
    
    # generate an example update (e.g., EMG singals that update exoskeleton joint torques as in [Peternel, 2016])
    U = 10*y # typically update factor is an input signal multiplied by a gain
    
    # set phase and period for DMPs
    myDMP.set_phase( np.array([phi]) )
    myDMP.set_period( np.array([tau]) )
    
    
    
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
    data.append([time,phi,x[0], y[0], x[0]-y[0]])
    data_robot.append(x.tolist())


data = np.asarray(data)
data_robot = np.asarray(data_robot)


sn = int(np.where(np.isclose(K1_2, K1max))[0][np.where(np.where(np.isclose(K1_2, K1max))[0] > (samples/2))[0][0]])



data_1_old = data_1[:sn]
data_robot_old = data_robot[:sn]

data_1_new = data_1[sn:]
data_robot_new = data_robot[sn:]
phase_new = phase[sn:]

K1_2_new = K1_2[sn:]
K1_2_new[K1_2_new <= 0] = 0

x2_last = data_robot_new[0,0]
x1_last = data_1_new[0,0]
dx1_last = data_1_new[0,1]

e_th = 0.1

data_1_new_ = []
data_2_new_ = []
time_new = []
K2_ = []
K2_last = 0


data_1_new_.append([x1_last, dx1_last])
data_2_new_.append([x2_last, dx1_last])
K2_.append(0)

for i in range(1,samples-sn):
    
    e_phi = data_robot_new[i,0] - x2_last
    
    if abs(e_phi) >= e_th:
        K2 = K2max
    else:
        K2 = 0
        
   
    K2_.append(K2)
    
    if np.sin(phase_new[i])>=0:
        ddx1 = K1_2_new[i]/m *(xr1l - x1_last) - (c1/m)*dx1_last + K2/m *(xr1l - x1_last) -  (c2/m)*dx1_last 
    else: 
        ddx1 = K1_2_new[i]/m *(xr1r - x1_last) - (c1/m)*dx1_last + K2/m *(xr1r - x1_last) -  (c2/m)*dx1_last 
        
    
    dx1 = ddx1*dt + dx1_last
    x1 = dx1*dt + x1_last
    
    data_1_new_.append([x1, dx1])
    data_2_new_.append([x1 + L, dx1])
    
    x1_last = x1
    dx1_last = dx1
    time_new.append(dt*i)
 
data_1_new_ = np.asarray(data_1_new_)
data_2_new_ = np.asarray(data_2_new_)


data_1 = np.concatenate((data_1_old, data_1_new_))
data_2 = np.concatenate((data_robot_old[:,0], data_2_new_[:,0]))
K1 = np.concatenate((K1_1[:sn], K1_2_new))
K2 = np.concatenate((np.zeros(sn), K2_))

fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios = [2,1]) 

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)

ax0.plot(time_[sn:],data_1_new_[:,0], 'b')
ax0.plot(time_[sn:], data_2_new_[:,0], 'r')
ax0.plot(time_[:sn], data_1_old[:,0], 'b')
ax0.plot(time_[:sn], data_robot_old[:,0], 'r')
ax0.axvline(x = time_[sn], color="grey")

ax1.plot(time_[:sn], K1_1[:sn], 'b')
ax1.plot(time_[:sn], np.zeros(sn), 'r')
ax1.plot(time_[sn:], K1_2_new, 'b')
ax1.plot(time_[sn:], K2_, 'r')  
plt.show()


fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios = [2,1]) 

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)

ax0.grid('on')
ax1.grid('on')

ax0.plot(time_,data_1[:,0])
ax0.plot(time_,data_2)
ax0.axvline(x = time_[sn], color="grey")

ax1.plot(time_, K1)
ax1.plot(time_, K2)
ax1.axvline(x = time_[sn], color="grey")
 
plt.show()


'''
    
plt.figure()
plt.plot(time_,K1_1, label = 'K1')
#plt.plot(time_,K2, label = 'K2')
plt.show()

    
plt.figure()
plt.plot(time_,data_1[:,0])
plt.plot(time_, data_2[:,0])
plt.plot(time_, data[:,2] )
plt.axvline(x = time_[int(samples/2)], color="grey")
plt.show()


plt.figure()
plt.plot(time_,data_1[:,1])
plt.show()


'''

if anim == 1:

    def animate(i):
        x1 = data_1[i,0]
        x2 = data_2[i]
        line.set_data([x1, x2], [0,0])
    
    def init():
        ax.set_xlim(-L-0.1,L + 0.1)
        ax.set_xlabel('$x$ [m]')
        line.set_data([L/2, -L/2],[0,0])
        return line
    
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set(yticklabels=[])  # remove the tick labels
    
    # These are the objects we need to keep track of.
    line, = ax.plot([], [], 'o-', lw=5, color='#de2d26')
    
    interval = 100*dt
    ani = animation.FuncAnimation(fig, animate, interval=interval, repeat=False, init_func=init)
    plt.show()
