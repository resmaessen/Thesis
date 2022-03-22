# -*- coding: utf-8 -*-

import os
from params import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec 
import learning_human_robot
import learning_robot_robot
import time
import params as p

plt.close('all')

time.sleep(2)

h = [p.h, 1, 5]
runs = 8


fig1, ax1 = plt.subplots(2,len(h), figsize=(20.0, 10.0), sharex=True)
fig2, ax2 = plt.subplots(runs,1, figsize=(20.0, 10.0), sharex=True)
fig3, ax3 = plt.subplots(runs,1, figsize=(20.0, 10.0), sharex=True)

dict_labels3 = {}


for i, h_ in enumerate(h):
    
    start = time.time()
    
    learning_human_robot.run_file(h=h_, v_h = True)
    learning_robot_robot.run_file(runs, h = h_, v_h = True)
    
    time_total = time.time()-start
    
    path = 'save_data'
    
    files = os.listdir(path)
    
    data_x_files = []
    data_k_files = []
    
    for f in files:
        if "_x" in f: data_x_files.append(f)
        elif "_k" in f: data_k_files.append(f)
        else: raise NameError('Other file')

        
    data_x_files.sort(key= lambda x: float(x.strip('data_').strip('_x')))
    data_k_files.sort(key= lambda x: float(x.strip('data_').strip('_k')))
    
    data_x_files = data_x_files[:runs]
    data_k_files = data_k_files[:runs]
    
        
    if len(data_x_files) != len(data_k_files):
        raise NameError('Data was saved incorrectly')
    
    data_x = []
    data_k = []
        
    for idx in range(len(data_x_files)):
        
        data_x.append(np.loadtxt("save_data/"+data_x_files[idx], unpack=True))
        data_k.append(np.loadtxt("save_data/"+data_k_files[idx], unpack=True))
        
    data_x = np.asarray(data_x)
    data_k = np.asarray(data_k)
    
    
    t = np.array([dt*i for i in range(samples)])
    
    
    ax1[0,i].set_title(r'h = '+str(h_))
    for idx in range(len(data_x_files)):
        if idx % 2 == 0:
            ax1[0,i].plot(t, data_x[idx,:,0], label = "run "+str(idx+1))
            ax1[1,i].plot(t, data_k[idx,:,0], label = "run "+str(idx+1))
    ax1[0,i].set_xticklabels([])
    ax1[1,i].set_xticklabels([])
    
    if i == 0:
        ax1[0,i].legend(loc = "lower right")
        ax1[0,i].set_ylabel('x[m]', fontsize='13')
        ax1[1,i].set_ylabel('K [N/m]', fontsize='13')
    ax1[1,i].set_xlabel('time [s]', fontsize='13')

    
    
    ax2[0].plot(t, data_x[0,:,0], label = str(h_))
    ax2[0].set_xticklabels([])
    ax2[0].set_ylabel('x0', fontsize = '13')
        
    for idx in range(1,len(data_x_files)):
        ax2[idx].plot(t, data_x[idx,:,0],label = r"h = " + str(h_)+ ' t = ' + str(time_total))
        ax2[idx].set_xticklabels([])
        ax2[idx].set_ylabel('x'+str(idx+1), fontsize='13')  
    
    if i == 0:
        ax2[-1].set_xlabel('time [s]', fontsize='13')

    
    ax3[0].plot(t, data_k[0,:,0], label = str(h_))
    ax3[0].set_xticklabels([])
    ax3[0].set_ylabel('K1', fontsize='13')
    
    
    for idx in range(1,len(data_k_files)):
        ax3[idx].plot(t, data_k[idx,:,0], label = r"h = " + str(h_)+ ' t = ' + str(time_total))
          
        ax3[idx].set_xticklabels([])
        ax3[idx].set_ylabel('K'+str(idx+1), fontsize='13')  
        #ax3[idx].legend(loc = 'center right')
    
    ax3[-1].set_xlabel('time [s]', fontsize='13')
    
        

    
ax2[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax3[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    
fig1.savefig('images/h/Skill_transfer_h_total.png')
fig2.savefig('images/h/Skill_transfer_h_total_x.png')
fig3.savefig('images/h/Skill_transfer_h_total_k.png')