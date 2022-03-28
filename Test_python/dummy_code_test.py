# -*- coding: utf-8 -*
import os
from params import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec 
from human_robot_weights import human_robot
from robot_robot_weights import robot_robot
import time
from pDMP_functions import pDMP

plt.close('all')
w_traject = np.loadtxt('save_data/data_w_traject_robot_1.csv', unpack=True)
w_stiff = np.loadtxt('save_data/data_w_stiff_robot_1.csv', unpack=True)

DMP_traject_exp = pDMP(DOF, N, alpha, beta, lambd, dt, h)
DMP_stiff_exp = pDMP(DOF, N, alpha, beta, lambd, dt, h)

if DOF == 1:
    DMP_traject_exp.set_weights(0, w_traject)
    DMP_stiff_exp.set_weights(0, w_stiff)

else:
    for i in range(DOF):
        DMP_stiff_exp.set_weights(i, w_traject[i,:])
        DMP_traject_exp.set_weights(i, w_stiff[i,:])

        
y_exp_last, dy_exp_last = np.array([0]), np.array([0])
x_exp, dy_exp = y_exp_last,  np.array([0])
k_exp =  np.array([0])




phase = [2*np.pi*(dt/tau)*i for i in range(samples*2)] 


# create a DMP object
DMP_traject_nov = pDMP(DOF, N, alpha, beta, lambd, dt, h)
DMP_stiff_nov = pDMP(DOF, N, alpha, beta, lambd, dt, h)


# Coefficients

y_nov_old = y_exp_last
dy_nov_old = dy_exp_last



data = []

learn = True

for i in range(int(samples)):
    # generate phase
    phi = phase[i]
    
    ddy_exp = k_exp/m*(x_exp-y_exp_last) - c1/m *dy_exp
    
    dy_exp = ddy_exp*dt + dy_exp_last
    y_exp = dy_exp*dt + y_exp_last
    
    # Get trajectory      
    DMP_stiff_exp.set_phase( np.array([phi]) )
    DMP_stiff_exp.set_period( np.array([tau]) )
    
    DMP_traject_exp.set_phase( np.array([phi]) )
    DMP_traject_exp.set_period( np.array([tau]) )
    
    
    
    if i < int(samples/3):
        # generate an example trajectory (e.g., the movement that is to be learned)
        y_nov = y_exp

        # calculate time derivatives
        dy_nov = (y_nov - y_nov_old) / dt 
        ddy_nov = (dy_nov - dy_nov_old) / dt
        
        K1 = K1max
        K2 = np.array([0])
        dK2 =  np.array([0])
        ddK2 = np.array([0])
        

        
    else: 
        K1 = k_exp[0]
        
        if i < int(samples*2/3):
                     
            if abs(x_nov - y_nov_old) >= e_th:
                K2 = np.array([K2max])
            else:
                K2 = np.array([0])
            dK2 = (K2 - K2_old) / dt 
            ddK2 = (dK2 - dK2_old) / dt
            
        else: 
            K2 = k_nov
        
        ddy_nov = K1/m * (x_exp-y_nov_old) - (c1/m)*dy_nov_old + k_nov[0]/m * (x_nov - y_nov_old) - (c2/m)*dy_nov_old
        dy_nov = ddy_nov*dt + dy_nov_old
        y_nov = dy_nov*dt + y_nov_old
        
    
    # set phase and period for DMPs
    DMP_traject_nov.set_phase( np.array([phi]) )
    DMP_traject_nov.set_period( np.array([tau]) )
    
    DMP_stiff_nov.set_phase( np.array([phi]) )
    DMP_stiff_nov.set_period( np.array([tau]) )
    
    # DMP mode of operation
    if i < int( 1/3 * samples ): # learn/update for half of the experiment time, then repeat that DMP until the end
        DMP_traject_nov.learn(y_nov, dy_nov, ddy_nov) # learn DMP based on a trajectory
        DMP_stiff_nov.repeat()
        stage = 1
    elif i < int( 2/3 * samples ):
        DMP_traject_nov.repeat() # repeat the learned DMP
        DMP_stiff_nov.learn(K2, dK2, ddK2)
        stage = 2
    else: 
        DMP_traject_nov.repeat()
        DMP_stiff_nov.repeat()
        stage = 3
        
    # DMP integration
    DMP_traject_nov.integration()
    DMP_stiff_nov.integration()
    
    
    DMP_traject_exp.repeat()
    DMP_stiff_exp.repeat()
    
    DMP_traject_exp.integration()
    DMP_stiff_exp.integration()
        
    
    
    
    # old values	
 
    x_exp, _, _ , _ = DMP_traject_exp.get_state()
    k_exp, _, _, _ = DMP_stiff_exp.get_state()
    
    
    y_exp_last, dy_exp_last = y_exp , dy_exp
    
    y_nov_old = y_nov
    dy_nov_old = dy_nov
    
    K2_old = K2
    dK2_old = dK2
    
    # store data for plotting
    x_nov, _, _, _ = DMP_traject_nov.get_state()
    k_nov, _, _, _ = DMP_stiff_nov.get_state()
    
    
    time = dt*i
    
    
    data.append([time,phi,stage,x_nov[0], y_nov[0], x_nov[0]-y_nov[0], k_nov[0], K1,  y_exp[0]])
    
        
data = np.asarray(data)



fig = plt.figure(figsize=(20.0, 10.0))
gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 1]) 

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)
ax2 = plt.subplot(gs[2], sharex = ax0)
ax3 = plt.subplot(gs[3], sharex = ax0)
ax4 = plt.subplot(gs[4])

ax0.plot(data[:,0], data[:,2])
ax1.plot(data[:,0], L-data[:,8], label = 'E ref')
ax1.plot(data[:,0], data[:,3], label = 'N res')
ax1.plot(data[:,0], data[:,4], label = 'N meas2')
ax2.plot(data[:,0], data[:,6])
ax3.plot(data[:,0], data[:,7])
ax4.plot(data[:,0], data[:,5])
ax4.axhline(y = e_th, color="black")
ax4.axhline(y = -e_th, color="black")

ax1.legend(loc = 'upper right')

ax0.set_xticklabels([])
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])


#ax2.set_ylim([0, K1max+100])
ax3.set_ylim([0, K1max+100])


ax0.set_ylabel('stage', fontsize='13')
ax1.set_ylabel('x [m]', fontsize='13')
ax2.set_ylabel('K [N/m]', fontsize='13')
ax3.set_ylabel('K [N/m]', fontsize='13')
ax4.set_ylabel('e [m]', fontsize='13')
ax4.set_xlabel('time [s]', fontsize='12')
