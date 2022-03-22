# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:51:01 2022

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
from params import *

runs = [2,3,4,5,6,7,8]

anim = 0
save_data = True


for run in runs:

    data_1= np.loadtxt("save_data/data_"+str(run-1)+"_x", unpack=True)
    data_2 = np.loadtxt("save_data/data_"+str(run-1)+"_x", unpack=True)
    data_k = np.loadtxt("save_data/data_"+str(run-1)+"_k", unpack=True)
    
    plt.close('all')
    
    K1, K2 = [], []
    phase = []
    
    K1_full = data_k[:,0]
    K1_full[:int(samples/3)] = K1max
    
    
    data_1[:,0] = data_1[:,0]-L
    
    y_old, dy_old = data_2[0,0], data_2[0,1]
    
    data = []
    
    # Coefficients
    x, dx = y_old, dy_old
    K2_old, dK2_old, k = 0, 0, 0
    phi = 0
    frequency_last = np.pi
    
    
    # create a DMP and AFS objects
    DMP_traject = pDMP(DOF, N, alpha, beta, lambd, dt)
    DMP_stiff = pDMP(DOF, N, alpha, beta, lambd, dt)
    
    myAFS = AFS(DOF, M, ni, K)
    myAFS.set_flag(1)
    myAFS.set_initial_AFS_state(frequency_last)
    
    if mode != 1:
        raise NameError('Inccorrect Mode')
    
    ''' Phase 1'''
    for i in range(int(samples)):
            
        
        if i < int(samples/3):
            # generate an example trajectory (e.g., the movement that is to be learned)
            y = np.array([data_2[i,0]])
            # calculate time derivatives
            dy = (y - y_old) / dt 
            ddy = (dy - dy_old) / dt
            
            K1 = K1_full[i]
            K2 = np.array([0])
            dK2 =  np.array([0])
            
            myAFS.update_input(0, np.array([y]))
            myAFS.AFS_integrate(dt, AFS_step)
            phi = myAFS.get_phase(0) 
            
        else:
            if i < int(samples*2/3): 
                         
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
                
            myAFS.set_flag(0)
            myAFS.AFS_integrate(dt, AFS_step)  
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
    
    
    fig = plt.figure(figsize=(20.0, 10.0))
    gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 1]) 
    
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex = ax0)
    ax2 = plt.subplot(gs[2], sharex = ax0)
    ax3 = plt.subplot(gs[3], sharex = ax0)
    ax4 = plt.subplot(gs[4])#, sharex = ax0)
    
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
    
    ax0.set_ylabel('stage', fontsize='13')
    ax1.set_ylabel('x [m]', fontsize='13')
    ax2.set_ylabel('K [N/m]', fontsize='13')
    ax3.set_ylabel('K [N/m]', fontsize='13')
    ax4.set_ylabel('e [m]', fontsize='13')
    ax4.set_xlabel('time [s]', fontsize='12')
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    # remove vertical gap between subplots
    plt.savefig('images/Skill_transfer_chain_'+str(run)+'.png')
    plt.close(fig)
    
    if save_data:
        data_x_save = []
        data_k_save = []
        myAFS.set_flag(0)
        
        for i in range(samples):
        
            
            myAFS.set_flag(0)
            myAFS.AFS_integrate(dt, AFS_step)  
            phi = myAFS.get_phase(0) 
            
            
            DMP_traject.set_phase( np.array([phi]) )
            DMP_traject.set_period( np.array([tau]) )
            
            DMP_stiff.set_phase( np.array([phi]) )
            DMP_stiff.set_period( np.array([tau]) )
            
            DMP_traject.repeat()
            DMP_stiff.repeat()
            
            DMP_traject.integration()
            DMP_stiff.integration()
            
            x, dx, ph, ta = DMP_traject.get_state()
            k, dk, phk, tak = DMP_stiff.get_state()
            
            y_old = y
            dy_old = dy
            
            data_x_save.append([x[0], dx[0]])
            data_k_save.append([k[0], dk[0]])
        
        data_x_save = np.asarray(data_x_save)
        data_k_save = np.asarray(data_k_save)
        
            
        file_x = open("save_data/data_"+str(run)+"_x", "w")
        np.savetxt(file_x, data_x_save.T , header="x y")
        file_k = open("save_data/data_"+str(run)+"_k", "w")
        np.savetxt(file_k, data_k_save.T, header = "k dk")
