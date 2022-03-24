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
from params import *
import random



def robot_robot(runs, e_th = e_th, N= N, h = h, \
             close = True, v_eth = False, v_N = False, v_h = False):
    
    for run in range(1,runs):

        w_traject_exp =  np.loadtxt('save_data/data_w_traject_robot_'+str(run)+'.csv', unpack = True)
        w_stiff_exp= np.loadtxt('save_data/data_w_stiff_robot_'+str(run)+'.csv', unpack = True)
        
        phase = [2*np.pi*(dt/tau)*i for i in range(samples*2)]
        
        K1_full = np.asarray([np.sin(phase[i]) * K1max for i in range(samples)])
        
        data_x_save = []
        data_k_save = []
        
        DMP_traject_exp = pDMP(DOF, N, alpha, beta, lambd, dt, h)
        DMP_stiff_exp = pDMP(DOF, N, alpha, beta, lambd, dt, h)
        
        if DOF == 1:
            DMP_traject_exp.set_weights(0, w_traject_exp)
            DMP_stiff_exp.set_weights(0, w_stiff_exp)
        else:
            for i in range(DOF):
                DMP_traject_exp.set_weights(i, w_traject_exp[i,:])
                DMP_stiff_exp.set_weights(i, w_stiff_exp[i,:])
        
        
        DMP_traject_exp.set_state(np.array([L]))
        
               
        y_old, dy_old = 0, 0
        x = y_old
        
        
        DMP_traject_nov = pDMP(DOF, N, alpha, beta, lambd, dt, h)
        DMP_stiff_nov = pDMP(DOF, N, alpha, beta, lambd, dt, h)
        
        phase = [2*np.pi*(dt/tau)*i for i in range(samples*2)]
        
        
        # Coefficients
        x, dx = y_old, dy_old
        K2_old, dK2_old, k = 0, 0, 0
        phi = 0
        frequency_last = np.pi
        
        
        data =  []
        
        if mode != 1:
            raise NameError('Inccorrect Mode')
        
        for i in range(int(samples)):
            # generate phase
            phi = phase[i]
            
            # Get trajectory      
            DMP_traject_exp.set_phase( np.array([phi]) )
            DMP_traject_exp.set_period( np.array([tau]) )
            
            DMP_stiff_exp.set_phase( np.array([phi]) )
            DMP_stiff_exp.set_period( np.array([tau]) )
            
            DMP_traject_exp.repeat()
            DMP_stiff_exp.repeat()
            
            DMP_traject_exp.integration()
            DMP_stiff_exp.integration()
            
            x_exp, _, _ , _ = DMP_traject_exp.get_state()
            k_exp, _, _, _ = DMP_stiff_exp.get_state()
            
                 
            if i < int(samples/3):
                      
                # generate an example trajectory (e.g., the movement that is to be learned)
                y = np.array([x_exp[0]-L])
                # calculate time derivatives
                dy = (y - y_old) / dt 
                ddy = (dy - dy_old) / dt
                
                K1 = K1max
                K2 = np.array([0])
                dK2 =  np.array([0])
                
            elif i < int(samples*2/3):
                             
                if abs(x - y_old) >= e_th:
                    K2 = np.array([K2max])
                else:
                    K2 = np.array([0])
                dK2 = (K2 - K2_old) / dt 
                ddK2 = (dK2 - dK2_old) / dt
                
                K1 = k_exp[0]
                
                F = K1* (x - y_old) - (c1)*dy_old + k * (x - y_old) - (c2)*dy_old
                
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
                K1 = k_exp[0]
                ddy = K1/m * (x - y_old) - (c1/m)*dy_old + k/m * (x - y_old) - (c2/m)*dy_old
                
                dy = ddy*dt + dy_old
                y = dy*dt + y_old
                
              
            # set phase and period for DMPs
            DMP_traject_nov.set_phase( np.array([phi]) )
            DMP_traject_nov.set_period( np.array([tau]) )
            
            DMP_stiff_nov.set_phase( np.array([phi]) )
            DMP_stiff_nov.set_period( np.array([tau]) )
            
            # DMP mode of operation
            if i < int( 1/3 * samples ): # learn/update for half of the experiment time, then repeat that DMP until the end
                DMP_traject_nov.learn(y, dy, ddy) # learn DMP based on a trajectory
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
            
            # old values	
            y_old = y
            dy_old = dy
            
            
            K2_old = K2
            dK2_old = dK2
            
            # store data for plotting
            x, dx, ph, ta = DMP_traject_nov.get_state()
            k, dk, phk, tak = DMP_stiff_nov.get_state()
            
            time = dt*i
            
            data.append([time,phi,stage,x[0], y[0], x[0]-y[0], k[0], K2[0],  x_exp[0], K1])
        
        
        data= np.asarray(data)
        
        w_traject_nov = np.zeros([DOF, N])
        w_stiff_nov = np.zeros([DOF, N])
        
        for i in range(DOF):
            w_traject_nov[i,:] = DMP_traject_nov.get_weights(i)
            w_stiff_nov[i,:] = DMP_stiff_nov.get_weights(i)
        
        
        fig = plt.figure(figsize=(20.0, 10.0))
        gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 1]) 
        
        
        
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax2 = plt.subplot(gs[2], sharex = ax0)
        ax3 = plt.subplot(gs[3], sharex = ax0)
        ax4 = plt.subplot(gs[4])#, sharex = ax0)
        
        ax0.plot(data[:,0], data[:,2])
        ax1.plot(data[:,0], -data[:,8], label = 'E ref')
        ax1.plot(data[:,0], data[:,3], label = 'N res')
        ax1.plot(data[:,0], data[:,4], label = 'N meas2')
        ax2.plot(data[:,0], data[:,6])
        ax3.plot(data[:,0], data[:,9])
        ax4.plot(data[:,0], data[:,5])
        ax4.axhline(y = e_th, color="black")
        ax4.axhline(y = -e_th, color="black")
        
        ax1.legend(loc = 'upper right')
        
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
        
        if v_eth: 
            fig.savefig('images/e_th/ST_e_'+str(e_th)+'_robot_'+str(run+1)+'.png')
        elif v_N:
            fig.savefig('images/N/ST_N_'+str(N)+'_robot_'+str(run+1)+'.png')
        elif v_h:
            fig.savefig('images/h/ST_h_'+str(h)+'_robot_'+str(run+1)+'.png')  
        else:
            fig.savefig('images/std/ST_std_robot_'+str(run+1)+' .png')
        
        np.savetxt('save_data/data_w_traject_robot_'+str(run+1)+'.csv', w_traject_exp)
        np.savetxt('save_data/data_w_stiff_robot_'+str(run+1)+'.csv', w_stiff_exp)

        if close:
            plt.close(fig)
        
        
if __name__ == "__main__":
    plt.close('all')
    robot_robot(2, close = False)
