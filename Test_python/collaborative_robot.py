# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:04:38 2022

@author: Rosa Maessen
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

plt.close('all')


# EXPERIMENT PARAMETERS
dt = 0.05 # system sample time
exp_time = 60 # total experiment time
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

for i in range(samples):
    
    phi += a*2*np.pi * dt/tau
    
    K1.append(np.sin(phi)*K1max) 
    K2.append(-np.sin(phi)*K2max)
    
    phase.append(phi)
    time_.append(dt*i)
    
K1 = np.asarray(K1)
K2 = np.asarray(K2)
K1[K1<=0] = 0
K2[K2<=0] = 0
    
plt.figure()
plt.plot(time_,K1, label = 'K1')
plt.plot(time_,K2, label = 'K2')
plt.show()

xm_init = 0
x1_init = 0 + L/2
x2_init = 0 - L/2

xr1l = x1_init -L/2 # Goal position
xr1r = x1_init +L/2
x1 = []


x1_last = x1_init
x2_last = x2_init

dx1_last = 0 

data_x1 = []
data_x2 = []
data_x1.append([x1_last, dx1_last])
data_x2.append([x2_last, dx1_last])

for i in range(1,samples):
    
    ddx = K1[i]/m *(xr1l - x1_last) - (c1/m)*dx1_last +  K2[i]/m *(xr1r - x1_last) -  (c2/m)*dx1_last 
    
    dx = ddx*dt + dx1_last
    x = dx*dt + x1_last
    
    data_x1.append([x, dx])
    data_x2.append([x - L, dx])
    
    x1_last = x
    dx1_last = dx
 
data_x1 = np.asarray(data_x1)
data_x2 = np.asarray(data_x2)
    
plt.figure()
plt.plot(time_,data_x1[:,0])
plt.plot(time_, data_x2[:,0])
plt.show()


plt.figure()
plt.plot(time_,data_x1[:,1])
plt.show()


''' Animiate Sawing '''

def animate(i):
    x1 = data_x1[i,0]
    x2 = data_x2[i,0]
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