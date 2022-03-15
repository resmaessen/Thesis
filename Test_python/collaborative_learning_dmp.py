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

phase = [2*np.pi*(dt/tau)*i for i in range(samples)]
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


y_old = data_2[0,0]
dy_old = 0

data = []

# create a DMP object
DMP_traject = pDMP(DOF, N, alpha, beta, lambd, dt)
DMP_stiff = pDMP(DOF, N, alpha, beta, lambd, dt)


# Coefficients
x, dx = y_old, dy_old

K2_old, dK2_old, k = 0, 0, 0

# Human to Robot

if mode != 1:
    raise NameError('Inccorrect Mode')
    

''' Phase 1'''
for i in range ( int(samples)):
    
    # generate phase
    phi = phase[i]
    
    
    if i < int(samples/3):
        # generate an example trajectory (e.g., the movement that is to be learned)
        y = np.array([data_2[i,0]])
        # calculate time derivatives
        dy = (y - y_old) / dt 
        ddy = (dy - dy_old) / dt
        
        K1 = K1_full[i]
        K2 = np.array([0])
        dK2 =  np.array([0])
        
    elif i < int(samples*2/3):
                     
        if abs(x - y_old) >= e_th:
            K2 = np.array([K2max])
        else:
            K2 = np.array([0])
        dK2 = (K2 - K2_old) / dt 
        ddK2 = (dK2 - dK2_old) / dt
        
        if c_manual:
            c1 = 2* m**0.5 * K1_full[i]**0.5
            c2 = 2* m**0.5 * np.abs(k[0])**0.5   
        F = K1_full[i] * (x - y_old) - (c1)*dy_old + k * (x - y_old) - (c2)*dy_old
        
        mu_s_ = mu_s + random.uniform(0,0.01)
        mu_k_ = mu_k + random.uniform(0,0.01)
            
        if abs(F) < mu_s_ * F_n:
            F = 0
        else:
            F = F - mu_k_*F_n*np.sign(F) 
            
        ddy = F/m
        dy = ddy*dt + dy_old
        y = dy*dt + y_old
    
    else: 
        if c_manual:
            c1 = 2* m**0.5 * K1_full[i]**0.5
            c2 = 2* m**0.5 * np.abs(k[0])**0.5    
        ddy = K1_full[i]/m * (x - y_old) - (c1/m)*dy_old + k/m * (x - y_old) - (c2/m)*dy_old
        
        dy = ddy*dt + dy_old
        y = dy*dt + y_old
        
    
    # set phase and period for DMPs
    DMP_traject.set_phase( np.array([phi]) )
    DMP_traject.set_period( np.array([tau]) )
    
    DMP_stiff.set_phase( np.array([phi]) )
    DMP_stiff.set_period( np.array([tau]) )
    
    # DMP mode of operation
    if i < int( 1/3 * samples ): # learn/update for half of the experiment time, then repeat that DMP until the end
        DMP_traject.learn(y, dy, ddy) # learn DMP based on a trajectory
        DMP_stiff.repeat()
        stage = 1
    elif i < int( 2/3 * samples ):
        DMP_traject.repeat() # repeat the learned DMP
        DMP_stiff.learn(K2, dK2, ddK2)
        stage = 2
    else: 
        DMP_traject.repeat()
        DMP_stiff.repeat()
        stage = 3
    
    # DMP integration
    DMP_traject.integration()
    DMP_stiff.integration()
    
    # old values	
    y_old = y
    dy_old = dy
    
    K2_old = K2
    dK2_old = dK2
    
    # store data for plotting
    x, dx, ph, ta = DMP_traject.get_state()
    k, dk, phk, tak = DMP_stiff.get_state()
    
    time = dt*i
    
    data.append([time,phi,stage,x[0], y[0], x[0]-y[0], k[0], K2[0], k[0]-K2[0]])


data= np.asarray(data)


fig = plt.figure()
gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 1]) 

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)
ax2 = plt.subplot(gs[2], sharex = ax0)
ax3 = plt.subplot(gs[3], sharex = ax0)
ax4 = plt.subplot(gs[4])

ax0.plot(data[:,0], data[:,2])
ax1.plot(data[:,0], -data_1[:,0], label = 'E ref')
ax1.plot(data[:,0], data[:,3], label = 'N res')
ax1.plot(data[:,0], data[:,4], label = 'N meas2')
ax2.plot(data[:,0], data[:,6])
ax3.plot(data[:,0], K1_full)
ax4.plot(data[:,0], data[:,5])
ax4.axhline(y = e_th, color="black")
ax4.axhline(y = -e_th, color="black")

ax1.legend(loc = 'upper left')

ax0.set_xticklabels([])
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])


ax2.set_ylim([0, K1max+100])
ax3.set_ylim([0, K1max+100])


ax0.set_ylabel('stage', fontsize='13')
ax1.set_ylabel('x [m]', fontsize='13')
ax2.set_ylabel('K [N/m]', fontsize='13')
ax3.set_ylabel('K [N/m]', fontsize='13')
ax4.set_ylabel('e [m]', fontsize='13')
ax4.set_xlabel('time [s]', fontsize='12')

# remove vertical gap between subplots
plt.show()
