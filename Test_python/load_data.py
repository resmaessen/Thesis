# -*- coding: utf-8 -*-

import os
from params import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec 

plt.close('all')

path = 'save_data'

files = os.listdir(path)

data_x_files = []
data_k_files = []

for f in files:
    if "_x" in f: data_x_files.append(f)
    elif "_k" in f: data_k_files.append(f)
    else: raise NameError('Other file')
    


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


fig = plt.figure(figsize=(18.0, 9.0))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)

for idx in range(len(data_x_files)):
    if idx % 2 == 0:
        ax0.plot(t, data_x[idx,:,0], label = "run "+str(idx+1))
        ax1.plot(t, data_k[idx,:,0], label = "run "+str(idx+1))
ax0.set_xticklabels([])
ax1.set_xticklabels([])
ax0.legend(loc = "lower right")
ax0.set_ylabel('x [m]', fontsize='13')
ax1.set_ylabel('K [N/m]', fontsize='13')
plt.savefig('images/Skill_transfer_total.png')
plt.show()



fig = plt.figure(figsize=(18.0, 9.0))
gs = gridspec.GridSpec(int(len(data_x_files)), 1, height_ratios=[1]*len(data_x_files)) 
ax0 = plt.subplot(gs[0])
ax0.plot(t, data_x[0,:,0], label = "run 1")
ax0.set_xticklabels([])
ax0.set_ylabel('x1', fontsize='13')
ax0.set_ylim([np.min(data_x[:,:,0]),np.max(data_x[:,:,0])])

for idx in range(1,len(data_x_files)):
    ax = plt.subplot(gs[idx], sharex = ax0)
    ax.plot(t, data_x[idx,:,0], label = "run "+str(idx+1))
    ax.set_xticklabels([])
    ax.set_ylabel('x'+str(idx+1), fontsize='13')  
    ax.set_ylim([np.min(data_x[:,:,0]),np.max(data_x[:,:,0])])

ax.set_xlabel('time [s]', fontsize='13')

plt.savefig('images/Skill_transfer_x.png')
plt.show()


fig = plt.figure(figsize=(18.0, 9.0))
gs = gridspec.GridSpec(int(len(data_x_files)), 1, height_ratios=[1]*len(data_k_files)) 
ax0 = plt.subplot(gs[0])
ax0.plot(t, data_k[0,:,0], label = "run 1")
ax0.set_xticklabels([])
ax0.set_ylabel('K1', fontsize='13')
ax0.set_ylim([np.min(data_k[:,:,0]),np.max(data_k[:,:,0])])

for idx in range(1,len(data_x_files)):
    ax = plt.subplot(gs[idx], sharex = ax0)
    ax.plot(t, data_k[idx,:,0], label = "run "+str(idx+1))
    ax.set_xticklabels([])
    ax.set_ylabel('K'+str(idx+1), fontsize='13')  
    ax.set_ylim([np.min(data_k[:,:,0]),np.max(data_k[:,:,0])])

ax.set_xlabel('time [s]', fontsize='13')

plt.savefig('images/Skill_transfer_k.png')
plt.show()














