# -*- coding: utf-8 -*
import os
from params import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from human_robot_weights import human_robot
from robot_robot_weights import robot_robot
from pDMP_functions import pDMP

from scipy.fft import rfft, rfftfreq

plt.close('all')


runs = 50
compute_data = False   
    

    
if compute_data:

    human_robot()
    robot_robot(runs)


path = 'save_data/std'

files = os.listdir(path)

files_w_traject = []
files_w_stiff = []

        

for f in files:
    if "data_std_w_traject_robot" in f: files_w_traject.append(f)
    elif "data_std_w_stiff_robot" in f: files_w_stiff.append(f)
    else: raise NameError('Other file')

    
files_w_traject.sort(key= lambda x: float(x.strip('data_std_w_traject_robot').strip('.csv')))
files_w_stiff.sort(key= lambda x: float(x.strip('data_std_w_stiff_robot').strip('.csv')))

files_w_traject = files_w_traject[:runs]
files_w_stiff = files_w_stiff[:runs]

    
if len(files_w_traject) != len(files_w_stiff):
    raise NameError('Data was saved incorrectly')

data_w_traject = []
data_w_stiff = []

p = 5
number_plot =round(runs/p)
   
fig1, ax1 = plt.subplots(2,1, figsize=(20.0, 10.0), sharex=True)
fig2, ax2 = plt.subplots(number_plot,1, figsize=(20.0, 10.0), sharex=True)
fig3, ax3 = plt.subplots(number_plot,1, figsize=(20.0, 10.0), sharex=True)
fig4, ax4 = plt.subplots(1,1, figsize=(20.0, 10.0), sharex=True)
   

t = np.array([dt*i for i in range(samples)])
phase = [2*np.pi*(dt/tau)*i for i in range(samples*2)] 

ax1[0].set_title('Std')

idx_plot = 0

for j in range(len(files_w_traject)):
    
    if j%p == 0: 
    
        w_traject = np.loadtxt('save_data/std/'+files_w_traject[j], unpack=True)
        w_stiff = np.loadtxt('save_data/std/'+files_w_stiff[j], unpack=True)
        
        DMP_traject = pDMP(DOF, N, alpha, beta, lambd, dt, h)
        DMP_stiff = pDMP(DOF, N, alpha, beta, lambd, dt, h)
        
        if DOF == 1:
            DMP_traject.set_weights(0, w_traject)
            DMP_stiff.set_weights(0, w_stiff)
        else:
            for i in range(DOF):
                DMP_traject.set_weights(i, w_traject[i,:])
                DMP_stiff.set_weights(i, w_stiff[i,:])
        
        
        DMP_traject.set_state(np.array([L]))
        
        data = []
        
        for idx in range(int(samples)):
            # generate phase
            phi = phase[idx]
            
            # Get trajectory      
            DMP_traject.set_phase( np.array([phi]) )
            DMP_traject.set_period( np.array([tau]) )
            
            DMP_stiff.set_phase( np.array([phi]) )
            DMP_stiff.set_period( np.array([tau]) )
            
            DMP_traject.repeat()
            DMP_stiff.repeat()
            
            DMP_traject.integration()
            DMP_stiff.integration()
            
            x, _, _ , _ = DMP_traject.get_state()
            k, _, _, _ = DMP_stiff.get_state()
            
            data.append([x[0], k[0]])
            
        data = np.asarray(data)
         
       
        if idx_plot % 2 == 0:
            ax1[0].plot(t, data[:,0], label = "run "+str(j+1))
            ax1[1].plot(t, data[:,1], label = "run "+str(j+1))
              
    
        
        ax2[idx_plot].plot(t, data[:,0])
        ax2[idx_plot].set_xticklabels([])
        ax2[idx_plot].set_ylabel('x'+str(j+1), fontsize='13')  
        
        ax3[idx_plot].plot(t, data[:,1])
        ax3[idx_plot].set_xticklabels([])
        ax3[idx_plot].set_ylabel('K'+str(j+1), fontsize='13')  
        
                   
        ax4.plot(rfftfreq(samples, dt), np.abs(rfft(data[:,1])), label = 'run '+str(j))
    
        idx_plot += 1

ax1[0].set_xticklabels([])
ax1[1].set_xticklabels([])

ax1[0].legend(loc = "lower right")
ax1[0].set_ylabel('x[m]', fontsize='13')
ax1[1].set_ylabel('K [N/m]', fontsize='13')

ax2[-1].set_xlabel('time [s]', fontsize='13')
ax3[-1].set_xlabel('time [s]', fontsize='13')
    
ax1[1].set_xlabel('time [s]', fontsize='13')

ax4.legend()
ax4.set_ylabel('Power')
ax4.set_xlabel('Frequency')
  
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()

save_data = True
if save_data: 
    fig1.savefig('images/STD/ST_std_total_long.png')
    fig2.savefig('images/STD/ST_std_total_x_long.png')
    fig3.savefig('images/STD/ST_std_total_k_long.png')
    fig4.savefig('images/STD/ST_std_frequency_long')
    
    fig1.savefig('images/final/ST_std_total_long.png')
    fig2.savefig('images/final/ST_std_total_x_long.png')
    fig3.savefig('images/final/ST_std_total_k_long.png')
    fig4.savefig('images/final/ST_std_frequency_long')
    


