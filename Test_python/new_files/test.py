# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from pDMP_functions import pDMP
from matplotlib import gridspec
import random
from params import *

plt.close('all')


file = 'data_full.txt'
data = []
ddata = []
time = []
time_p = []

total = 18

with open(file,'r') as infile:
    for idx, line in enumerate(infile):
        if (idx-14)%(total) == 0:
            data.append(line)
        if (idx-15)%(total) == 0:
            ddata.append(line)
        elif (idx-3)%(total) ==0:
            time.append(line)
        elif (idx-4)%(total) == 0:
            time_p.append(line)

data = np.asarray(data)
ddata = np.asarray(ddata)

DOF = 7
data_correct = np.zeros([len(data)-1, DOF])
ddata_correct = np.zeros([len(ddata)-1, DOF])
time_real = np.zeros([len(time)-1])

for i in range(len(data)-1):
    for j in range(DOF):
        data_correct[i, j] = float(data[i+1].strip('position:')[2:-2].split(',', 7)[j])
        ddata_correct[i, j] = float(ddata[i+1].strip('velocity:')[2:-2].split(',', 7)[j])
    time_real[i] = float(time[i+1].strip(' secs:')) + float(time_p[i+1].strip(' nsecs:'))*10**-9

'''
plt.figure()
for i in range(DOF):
    plt.plot(time_real, data_correct[:,i], label = 'real'+ str(i))
    plt.legend()

plt.show()

'''
dt_ = []

for k in range(len(time_real)-1):
    dt_.append(np.abs(time_real[k+1]-time_real[k]))

DOF = 1

phi = 0

samples = len(data_correct)-1

data_save = []
data_x = []

n = 1

y_old = data_correct[0,n]
dy_old = ddata_correct[0,n]

DMP_traject = pDMP(DOF, N, alpha, beta, lambd, dt_[0], h)



time = 0

for i in range(samples):

    dt = dt_[i]

    # generate an example trajectory (e.g., the movement that is to be learned)
    y = np.array([data_correct[i,n]])
    # calculate time derivatives
    dy = (y - y_old) / dt
    ddy = (dy - dy_old) / dt
    
    

    # set phase and period for DMPs
    DMP_traject.set_phase( np.array([phi]))#, phi, phi, phi, phi, phi, phi]) )
    DMP_traject.set_period( np.array([tau]))#, tau, tau, tau, tau, tau, tau]) )
    DMP_traject.set_dt(dt)


    # DMP mode of operation
    if i < int( 1/2 * samples ): # learn/update for half of the experiment time, then repeat that DMP until the end
        DMP_traject.learn(y, dy, ddy) # learn DMP based on a trajectory
        stage = 1
    else:
        DMP_traject.repeat()

    # DMP integration
    DMP_traject.integration()


    # old values
    y_old = y
    dy_old = dy

    
    # store data for plotting
    x, dx, ph, ta = DMP_traject.get_state()
    

    time += dt

    data_save.append([time,phi,stage])
    data_x.append([x[0], dx[0], y[0]])#, x[1], x[2], x[3], x[4], x[5], x[6]])



    phi += 2*np.pi*(dt/(tau))

data_save = np.asarray(data_save)
data_x = np.asarray(data_x)

plt.figure()
plt.plot(data_save[:,0], data_x[:,0], label = 'enc')
plt.plot(data_save[:,0], data_correct[:-1,n], label = 'real')
plt.plot(data_save[:,0], data_x[:,2])
plt.legend()

plt.show()
'''
plt.figure()
for i in range(DOF):
    plt.plot(data_save[:,0], data_x[:,i], label = 'enc' + str(i))
    if DOF == 1:
        i = n
    plt.plot(data_save[:,0], data_correct[:-1,i], label = 'real'+ str(i))
    plt.legend()

plt.show()
'''

plt.figure()
plt.plot(data_save[:,0], data_x[:,1])