# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:04:38 2022

@author: Rosa Maessen
"""

import numpy as np
import matplotlib.pyplot as plt
from pDMP_functions import pDMP
from matplotlib import gridspec 
import random
from params import *



plt.close('all')
    
    
anim = 0

c_manual = False

K1 = []
K2 = []
phase = []

phase = [2*np.pi*(dt/tau)*i for i in range(samples*2)]

K1_full = np.asarray([np.sin(phase[i]) * K1max for i in range(samples)])

K1_full[:int(samples/3)] = K1max

K1_full[K1_full<0] = 0

xm_init = 0
x1_init = xm_init - L/2
x2_init = xm_init + L/2

xr1l = x1_init -L/2 # Goal position
xr1r = x1_init +L/2

xr2l = xr1l + L
xr2r = xr1r + L

x1_last = x1_init

dx1_last = 0 
dx2_last = dx1_last

data_1 = []
data_2 = []
data_1.append([x1_last, dx1_last])
data_2.append([x2_init, dx2_last])


F_save = []
F_save.append(0)


for i in range(1,samples):
    if c_manual:
        c1 = 2* m**0.5 * K1max**0.5
    if np.sin(phase[i])>=0:
        F = K1max*(xr1l - x1_last) - c1*dx1_last
    else: 
        F = K1max*(xr1r - x1_last) - c1*dx1_last
        
    mu_s_ = mu_s + random.uniform(0,0.01)
    mu_k_ = mu_k + random.uniform(0,0.01)
        
    if abs(F) < mu_s_ * F_n:
        F = 0
    else:
        F = F - mu_k_*N*np.sign(F) 
        
    ddx = F/m
        
    dx = ddx*dt + dx1_last
    x = dx*dt + x1_last
    
    data_1.append([x, dx])
    data_2.append([x + L, dx])
    F_save.append(F)
    
    x1_last = x
    dx1_last = dx
    
data_1 = np.asarray(data_1)
data_2 = np.asarray(data_2)
data_F = np.asarray(F_save)


y_old = data_F[0]
dy_old = 0

data = []

h = 0.5

# create a DMP object
myDMP = pDMP(DOF, N, alpha, beta, lambd, dt, h)


# Coefficients
x, dx = y_old, dy_old

F_old, dF_old = 0, 0


# Human to Robot

if mode != 1:
    raise NameError('Inccorrect Mode')
    

''' Phase 1'''
for i in range ( int(samples)):
    
    # generate phase
    phi = phase[i]
    
    if i < int(samples/2):
        if abs(x - y_old) >= e_th:
                K2 = np.array([K2max])
        else:
            K2 = np.array([0])
                
        F = K1_full[i] * (data_1[i,0] - y_old) - (c1)*dy_old + K2 * (data_1[i,0] - y_old) - (c2)*dy_old
        
        u_s_ = mu_s + random.uniform(0,0.01)
        mu_k_ = mu_k + random.uniform(0,0.01)
            
        if abs(F) < mu_s_ * F_n:
            F = 0
        else:
            F = F - mu_k_*F_n*np.sign(F) 
            
        F = np.array([F])
            
        ddy = F/m
        dy = ddy*dt + dy_old
        y = dy*dt + y_old
    
    else:
        ddy = x/m
        dy = ddy*dt+dy_old
        y = dy*dt + y_old
    

    # generate an example trajectory (e.g., the movement that is to be learned)
    #y = np.array([data_F[i]])
    # calculate time derivatives
    dF = (F - F_old) / dt 
    ddF = (dF - dF_old) / dt
    
    # generate an example update (e.g., EMG singals that update exoskeleton joint torques as in [Peternel, 2016])
    U = 10*y # typically update factor is an input signal multiplied by a gain
    
    # set phase and period for DMPs
    myDMP.set_phase( np.array([phi]) )
    myDMP.set_period( np.array([tau]) )
    
    
    
    # DMP mode of operation
    if i < int( 0.5 * samples ): # learn/update for half of the experiment time, then repeat that DMP until the end
        if( mode == 1 ):
            myDMP.learn(F, dF, ddF) # learn DMP based on a trajectory
        elif ( mode == 2 ):
            myDMP.update(U) # update DMP based on an update factor
    else:
        myDMP.repeat() # repeat the learned DMP
    
    # DMP integration
    myDMP.integration()
    
    
    # old values	
    y_old = y
    dy_old = dy
    
    F_old = F
    dF_old = dF
    
    # store data for plotting
    x, dx, ph, ta = myDMP.get_state()
    time = dt*i
    
    data.append([time,phi,x[0],F[0], y])
    
data= np.asarray(data)


plt.figure()
plt.plot(data[:,0], data_F)
plt.plot(data[:,0], data[:,3])
plt.plot(data[:,0], data[:,2])

plt.figure()
plt.plot(data[:,0], data_1[:,0])
plt.plot(data[:,0], data[:,4])
'''
datab = []
x1_last = 0
dx1_last = 0

for i in range(int(samples/2),int(samples)):
    ddx = data[i,2]/m
        
    dx = ddx*dt + dx1_last
    x = dx*dt + x1_last
    
    datab.append([x, dx])
    
    x1_last = x
    dx1_last = dx
    
datab = np.asarray(datab)



plt.figure()
plt.plot(data[:,0], data_F, label = 'ref')
plt.plot(data[:,0], data[:,2], label = 'encoded')
plt.legend()

plt.figure()
plt.plot(data[int(samples/2):,0], data_1[int(samples/2):,0])
plt.plot(data[int(samples/2):,0], datab[:,0])

plt.figure()
plt.plot(data[int(samples/2):,0], data_1[int(samples/2):,1])
plt.plot(data[int(samples/2):,0], datab[:,1])
'''