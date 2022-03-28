# -*- coding: utf-8 -*
import os
from params import *
import numpy as np
import matplotlib.pyplot as plt
from human_robot_weights import human_robot
from robot_robot_weights import robot_robot
import time
from pDMP_functions import pDMP

plt.close('all')

time.sleep(2)


runs = 20

running = np.array([[False, False, True]])



v_N, v_e, v_h  = running[0]
h = 2.5
name = 'h'

v = h

fig1, ax1 = plt.subplots(2,1, figsize=(20.0, 10.0), sharex=True)
fig2, ax2 = plt.subplots(runs,1, figsize=(20.0, 10.0), sharex=True)
fig3, ax3 = plt.subplots(runs,1, figsize=(20.0, 10.0), sharex=True)


start = time.time()

human_robot(N= N, e_th = e_th, h = h, v_N = v_N, v_eth = v_e, v_h = v_h)
robot_robot(runs, N= N, e_th = e_th, h = h, v_N = v_N, v_eth = v_e, v_h = v_h)

time_total = time.time()-start

path = 'save_data'

files = os.listdir(path)

files_w_traject = []
files_w_stiff = []

        

for f in files:
    if "data_w_traject_robot" in f: files_w_traject.append(f)
    elif "data_w_stiff_robot" in f: files_w_stiff.append(f)
    else: raise NameError('Other file')

    
files_w_traject.sort(key= lambda x: float(x.strip('data_w_traject_robot').strip('.csv')))
files_w_stiff.sort(key= lambda x: float(x.strip('data_w_stiff_robot').strip('.csv')))

files_w_traject = files_w_traject[:runs]
files_w_stiff = files_w_stiff[:runs]

    
if len(files_w_traject) != len(files_w_stiff):
    raise NameError('Data was saved incorrectly')

data_w_traject = []
data_w_stiff = []
    
   

t = np.array([dt*i for i in range(samples)])
phase = [2*np.pi*(dt/tau)*i for i in range(samples*2)] 

ax1[0].set_title(name +str(N))

for j in range(len(files_w_traject)):
    
    w_traject = np.loadtxt('save_data/'+files_w_traject[j], unpack=True)
    w_stiff = np.loadtxt('save_data/'+files_w_stiff[j], unpack=True)
    
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
   
    if j % 4 == 0:
        ax1[0].plot(t, data[:,0], label = "run "+str(j+1))
        ax1[1].plot(t, data[:,1], label = "run "+str(j+1))
        

    
    ax2[j].plot(t, data[:,0],label = name +' = ' + str(v) +'t = ' + str(time_total))
    ax2[j].set_xticklabels([])
    ax2[j].set_ylabel('x'+str(j+1), fontsize='13')  
    
    ax3[j].plot(t, data[:,1],label = name +' = ' + str(v) +'t = ' + str(time_total))
    ax3[j].set_xticklabels([])
    ax3[j].set_ylabel('K'+str(j+1), fontsize='13')  

       

ax1[0].set_xticklabels([])
ax1[1].set_xticklabels([])

ax1[0].legend(loc = "lower right")
ax1[0].set_ylabel('x[m]', fontsize='13')
ax1[1].set_ylabel('K [N/m]', fontsize='13')
ax1[1].set_xlabel('time [s]', fontsize='13')

ax2[-1].set_xlabel('time [s]', fontsize='13')
ax3[-1].set_xlabel('time [s]', fontsize='13')
    


  

    

ax2[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax3[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))


fig1.savefig('images/test/ST_'+name+'_total.png')
fig2.savefig('images/test/ST_'+name+'_total_x.png')
fig3.savefig('images/test/ST_'+name+'_total_k.png')








