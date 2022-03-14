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
from AFS_structure import AFS
from IPython.display import HTML
import random

plt.close('all')

anim = 0

# EXPERIMENT PARAMETERS
dt = 0.05 # system sample time
exp_time = 30 # total experiment time
samples = int(1/dt) * exp_time

DOF = 1 # degrees of freedom (number of DMPs to be learned)
N = 25 # number of weights per DMP (more weights can reproduce more complicated shapes)
alpha = 8 # DMP gain alpha
beta = 2 # DMP gain beta
lambd = 0.995 # forgetting factor
tau = 5 # DMP time period = 1/frequency (NOTE: this is the frequency of a periodic DMP)
phi = 0 # DMP phase

mode = 1 # DMP mode of operation (see below for details)

tau = 1/0.5 # Time period = 1/frequency (NOTE: this is the frequency of a period)
phi = 0 # Starting phase

L = 0.15  # Length movement

K1max = 1100 # N/m
K2max = 1100 # N/m

c1 = 110
c2 = 110

m = 10 # kg
g = 9.81

mu_s = 0.3
mu_k = 0.2

e_th = 0.02

ni = 2
K = 20
M = 10

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

F_y = 5

F_n = m*g + F_y

F_fs_max = mu_s * N
F_fk = mu_k*N

F_save = []


for i in range(1,samples):
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

phi = 0


frequency_last = np.pi

myAFS = AFS(DOF, M, ni, K)
myAFS.set_flag(1)
myAFS.set_initial_AFS_state(frequency_last)

if mode != 1:
    raise NameError('Inccorrect Mode')
    

''' Phase 1'''
for i in range ( int(samples)):
    
    # generate phase
    #phi = phase[i]
    
    
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
        
        ddy = K1_full[i]/m * (x - y_old) - (c1/m)*dy_old + k/m * (x - y_old) - (c2/m)*dy_old
        
        dy = ddy*dt + dy_old
        y = dy*dt + y_old
       
    myAFS.update_input(0, np.array([y]))
    myAFS.AFS_integrate(dt, 10)  
    
    phi = myAFS.get_phase(0) 
    
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
    
phi_ = np.mean(data[int(samples*2/3):,2])

phase_save = [phi_*np.pi*(dt/tau)*i for i in range(samples)]
data_x = []
data_k = []

for i in range(samples):
    phi = phase_save[i]+data[-1,1]
    
    DMP_traject.set_phase( np.array([phi]) )
    DMP_traject.set_period( np.array([tau]) )
    
    DMP_stiff.set_phase( np.array([phi]) )
    DMP_stiff.set_period( np.array([tau]) )
    
    DMP_traject.repeat()
    DMP_stiff.repeat()
    
    DMP_traject.integration()
    DMP_stiff.integration()
    
    # store data for plotting
    x, dx, ph, ta = DMP_traject.get_state()
    k, dk, phk, tak = DMP_stiff.get_state()
    
    data_x.append([x, dx])
    data_k.append([k,dk])




fig = plt.figure()
gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 1]) 

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)
ax2 = plt.subplot(gs[2], sharex = ax0)
ax3 = plt.subplot(gs[3], sharex = ax0)
ax4 = plt.subplot(gs[4])#, sharex = ax0)

ax0.plot(data[:,0], data[:,2])
ax1.plot(data[:,0], -data_1[:,0], label = 'E ref')
ax1.plot(data[:,0], data[:,3], label = 'test')
ax1.plot(data[:,0], data[:,4], label = 'test 2')
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

ax0.set_ylabel('stage', fontsize='13')
ax1.set_ylabel('x [m]', fontsize='13')
ax2.set_ylabel('K [N/m]', fontsize='13')
ax3.set_ylabel('K [N/m]', fontsize='13')
ax4.set_ylabel('e [m]', fontsize='13')
ax4.set_xlabel('time [s]', fontsize='12')

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# remove vertical gap between subplots
plt.savefig('images/Skill_transfer_chain.png')
plt.show()

plt.figure()
plt.plot(data[:,0], data[:,1], label = 'estimate')
plt.plot(data[:,0], phase, label = 'real')
plt.ylabel('phase [rad]')
plt.xlabel('time [s]')
plt.show()

TRY TO FIX THIS

''' Save the data '''

phi_ = np.mean(data[int(samples*2/3):,2])

phase_save = [phi_*np.pi*(dt/tau)*i for i in range(samples)]
data_x = []
data_k = []

for i in range(samples):
    phi = phase_save[i]+data[-1,1]
    
    DMP_traject.set_phase( np.array([phi]) )
    DMP_traject.set_period( np.array([tau]) )
    
    DMP_stiff.set_phase( np.array([phi]) )
    DMP_stiff.set_period( np.array([tau]) )
    
    DMP_traject.repeat()
    DMP_stiff.repeat()
    
    DMP_traject.integration()
    DMP_stiff.integration()
    
    # store data for plotting
    x, dx, ph, ta = DMP_traject.get_state()
    k, dk, phk, tak = DMP_stiff.get_state()
    
    data_x.append([x, dx])
    data_k.append([k,dk])
    


data_x = np.asarray(data_x)[:,:,0]
data_k = np.asarray(data_k)[:,:,0]

    
file_x = open("save_data/data_x", "w")
np.savetxt(file_x, data_x.T , header="x y")
file_k = open("save_data/data_k", "w")
np.savetxt(file_k, data_k.T, header = "k dk")

