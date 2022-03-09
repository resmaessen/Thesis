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
from IPython.display import HTML

plt.close('all')

anim = 1

# EXPERIMENT PARAMETERS
dt = 0.05 # system sample time
exp_time = 100 # total experiment time
samples = int(1/dt) * exp_time

tau = 5 # Time period = 1/frequency (NOTE: this is the frequency of a period)
phi = 0 # Starting phase

L = 0.5 # Length saw

K1max = 100 # N/m
K2max = 100 # N/m

c1 = 50
c2 = 50

m = 10 # kg

K1 = []
K2 = []
time_ = []
phase = []

a = 0.5

phase = [a * 2*np.pi*(dt/tau)*i for i in range(samples)]
K1_1 = np.asarray([K1max for i in range(samples)])
K1_2 = np.asarray([np.sin(phase[i]) * K1max for i in range(samples)])

xm_init = 0
x1_init = xm_init - L/2
x2_init = xm_init + L/2

xr1l = x1_init -L/2 # Goal position
xr1r = x1_init +L/2

x1_last = x1_init

dx1_last = 0 
dx2_last = dx1_last

data_1 = []
data_2 = []
data_1.append([x1_last, dx1_last])
data_2.append([x2_init, dx2_last])

time_.append(0)

for i in range(1,samples):
    
    
    if np.sin(phase[i])>=0:
        ddx = K1_1[i]/m *(xr1l - x1_last) - (c1/m)*dx1_last # +  K2[i]/m *(xr1r - x1_last) -  (c2/m)*dx1_last 
    else: 
        ddx = K1_1[i]/m * (xr1r-x1_last) -  (c1/m)*dx1_last 
        
    dx = ddx*dt + dx1_last
    x = dx*dt + x1_last
    
    data_1.append([x, dx])
    data_2.append([x + L, dx])
    
    x1_last = x
    dx1_last = dx
    time_.append(dt*i)
 
data_1 = np.asarray(data_1)
data_2 = np.asarray(data_2)


DOF = 1 # degrees of freedom (number of DMPs to be learned)
N = 25 # number of weights per DMP (more weights can reproduce more complicated shapes)
alpha = 8 # DMP gain alpha
beta = 2 # DMP gain beta
lambd = 0.995 # forgetting factor
tau = 5 # DMP time period = 1/frequency (NOTE: this is the frequency of a periodic DMP)
phi = 0 # DMP phase

mode = 1 # DMP mode of operation (see below for details)

y_old = data_2[0,0]
dy_old = 0

data = []
data_2_ = []

# create a DMP object
myDMP = pDMP(DOF, N, alpha, beta, lambd, dt)


# Coefficients
a, b = 1, 0

# Human to Robot
for i in range ( samples ):

    # generate phase
    phi = phase[i]
    
    # generate an example trajectory (e.g., the movement that is to be learned)
    y = np.array([data_2[i,0]])
    # calculate time derivatives
    dy = (y - y_old) / dt 
    ddy = (dy - dy_old) / dt
    
    # generate an example update (e.g., EMG singals that update exoskeleton joint torques as in [Peternel, 2016])
    U = 10*y # typically update factor is an input signal multiplied by a gain
    
    # set phase and period for DMPs
    myDMP.set_phase( np.array([phi]) )
    myDMP.set_period( np.array([tau]) )
    
    # DMP mode of operation
    if i < int( 1/2 * samples ): # learn/update for half of the experiment time, then repeat that DMP until the end
        if( mode == 1 ):
            myDMP.learn(y, dy, ddy) # learn DMP based on a trajectory
        elif ( mode == 2 ):
            myDMP.update(U) # update DMP based on an update factor
    else:
        myDMP.repeat() # repeat the learned DMP
    
    # DMP integration
    myDMP.integration()
    
    
    # old values	
    y_old = y
    dy_old = dy
    
    # store data for plotting
    x, dx, ph, ta = myDMP.get_state()
    time = dt*i
    data.append([time,phi,x[0], y[0], x[0]-y[0]])
    data_2_.append(x.tolist())


data = np.asarray(data)
data_2_ = np.asarray(data_2_)


sn = int(np.where(np.isclose(K1_2, K1max))[0][np.where(np.where(np.isclose(K1_2, K1max))[0] > (samples*1/2))[0][0]])



data_1_learn = data_1[:sn]
data_2_learn = data_2_[:sn]

#data_1_repro = data_1[sn:]
data_2_repro_ex = data_2_[sn:]
phase_new = phase[sn:]

K1_2_new = K1_2[sn:]
K1_2_new[K1_2_new <= 0] = 0


x1_last = data_1[sn+1,0]
x2_last = data_2[sn+1,0]
dx1_last = data_1[sn+1,1]


e_th = .1
data_1_repro = []
data_2_repro = []
time_new = []
K2_learn = []
K2_last = 0


data_1_repro.append([x1_last, dx1_last])
data_2_repro.append([x2_last, dx1_last])
K2_learn.append(K2_last)

for i in range(1,samples-sn):
    
    e_phi = data_2_repro_ex[i,0] - x2_last
    
    if abs(e_phi) >= e_th:
        K2 = K2max
    else:
        K2 = 0
        
    K2_learn.append(K2)
    
    if np.sin(phase_new[i])>=0:
        ddx1 = K1_2_new[i]/m *(xr1l - x1_last) - (c1/m)*dx1_last + K2/m *(xr1l - x1_last) -  (c2/m)*dx1_last 
    else: 
        ddx1 = K1_2_new[i]/m *(xr1r - x1_last) - (c1/m)*dx1_last + K2/m *(xr1r - x1_last) -  (c2/m)*dx1_last 
        
    
    dx1 = ddx1*dt + dx1_last
    x1 = dx1*dt + x1_last
    
    data_1_repro.append([x1, dx1])
    data_2_repro.append([x1 + L, dx1])
    
    x1_last = x1
    dx1_last = dx1
    time_new.append(dt*i)
 
data_1_repro = np.asarray(data_1_repro)
data_2_repro = np.asarray(data_2_repro)

data_1_total = np.concatenate((data_1_learn[:,0], data_1_repro[:,0]))
#data_2_total = np.concatenate((data_2_learn[:,0], data_2_repro[:,0]))
data_2_total = np.concatenate((data_2[:sn,0], data_2_repro[:,0]))
K1 = np.concatenate((K1_1[:sn], K1_2_new))
K2 = np.concatenate((np.zeros(sn), K2_learn))




''' Figures '''
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios = [2,1]) 

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)

ax0.grid('on')
ax1.grid('on')

ax0.plot(time_,data_1_total)
ax0.plot(time_,data_2_total)
ax0.plot(time_, data_2_[:,0], '--')
ax0.axvline(x = time_[sn], color="grey")

ax1.plot(time_, K1)
ax1.plot(time_, K2)
ax1.axvline(x = time_[sn], color="grey")

ax0.set_ylabel('position [m]')
ax1.set_ylabel('Stiffnes [N/m]')
ax1.set_xlabel('time [s]')
 
plt.savefig('images/stifness_e_th='+str(e_th)+'.png')
plt.show()


''' Learn something new ''' 

data_2b = data_2_total[sn:] - L


for idx in range(len(data_2b)-1):
    if abs(data_2b[idx] - data_2b[idx+1]) > 0.001:
        break

for idx2 in range(50,len(data_2b)-1-idx):
    if abs(data_2b[len(data_2b)-1-idx2] - data_2b[len(data_2b)-2-idx2]) < 0.001:
        break

sn2 = sn + idx
data_2b = data_2b[idx:len(data_2b)-idx2] 


    
data_3 = data_2b + L

data_3_long = np.concatenate((data_3, data_3))

K2b = K2[sn2:len(data_2b)-idx2]

y_old = data_3_long[0]
dy_old = data_2_repro[0]

datab = []
data_3_ = []

# create a DMP object
myDMP2 = pDMP(DOF, N, alpha, beta, lambd, dt)

samples2 = len(data_3_long)
# Human to Robot
for i in range ( samples2):

    # generate phase
    phi = phase[i]
    
    # generate an example trajectory (e.g., the movement that is to be learned)
    y = np.array([data_3_long[i]])
    # calculate data_3_long derivatives
    dy = (y - y_old) / dt 
    ddy = (dy - dy_old) / dt
    
    # generate an example update (e.g., EMG singals that update exoskeleton joint torques as in [Peternel, 2016])
    U = 10*y # typically update factor is an input signal multiplied by a gain
    
    # set phase and period for DMPs
    myDMP2.set_phase( np.array([phi]) )
    myDMP2.set_period( np.array([tau]) )
    
    # DMP mode of operation
    if i < int( 1/2 * samples2 ): # learn/update for half of the experiment time, then repeat that DMP until the end
        if( mode == 1 ):
            myDMP2.learn(y, dy, ddy) # learn DMP based on a trajectory
        elif ( mode == 2 ):
            myDMP2.update(U) # update DMP based on an update factor
    else:
        myDMP2.repeat() # repeat the learned DMP
    
    # DMP integration
    myDMP2.integration()
    
    
    # old values	
    y_old = y
    dy_old = dy
    
    # store data for plotting
    x, dx, ph, ta = myDMP2.get_state()
    time = dt*i
    datab.append([time,phi,x[0], y[0], x[0]-y[0]])
    data_3_.append(x.tolist())


datab = np.asarray(datab)
data_3_ = np.asarray(data_3_)


## NEW
data_3_learn = data_3_[:int(samples2/2)]

#data_1_repro = data_1[sn:]
data_3_repro_ex = data_3_[int(samples2/2):]
phaseb_new = phase[int(samples2/2):]



x2_last = data_2b[int(samples2/2)-1]
x3_last = data_3_learn[int(samples2/2)-1]
dx1_last = data_2_repro[0]


e_th = .1
data_2b_repro = []
data_3_repro = []
time_new = []
K3_learn = []
K3_last = 0


data_2b_repro.append([x2_last, dx2_last])
data_3_repro.append([x3_last, dx2_last])
K3_learn.append(K3_last)

for i in range(1,int(samples2/2)):
    
    e_phi = data_3_repro_ex[i,0] - x3_last
    
    if abs(e_phi) >= e_th:
        K3 = K2max
    else:
        K3 = 0
        
    K3_learn.append(K3)
    
    if np.sin(phase_new[i])>=0:
        ddx3 = K2b[i]/m *(xr1l - x2_last) - (c1/m)*dx1_last + K3/m *(xr1l - x2_last) -  (c2/m)*dx2_last 
    else: 
        ddx2 = K2b[i]/m *(xr1r - x2_last) - (c1/m)*dx1_last + K3/m *(xr1r - x2_last) -  (c2/m)*dx2_last 
        
    
    dx2 = ddx2*dt + dx2_last
    x2 = dx2*dt + x2_last
    
    data_2_repro.append([x2, dx2])
    data_3_repro.append([x2 + L, dx2])
    
    x2_last = x2
    dx2_last = dx2
    time_new.append(dt*i)
 
data_2_repro = np.asarray(data_2_repro)
data_3_repro = np.asarray(data_3_repro)

data_2_total = np.concatenate((data_2_learn[:,0], data_2_repro[:,0]))
#data_2_total = np.concatenate((data_2_learn[:,0], data_2_repro[:,0]))
data_3_total = np.concatenate((data_3[:int(samples2/2),0], data_3_repro[:,0]))
#K1 = np.concatenate((K1_1[:sn], K1_2_new))
#K2 = np.concatenate((np.zeros(sn), K2_learn))


''' Figures '''
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios = [2,1]) 

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)

ax0.grid('on')
ax1.grid('on')

ax0.plot(time_,data_2_total)
ax0.plot(time_,data_3_total)
ax0.plot(time_, data_3_[:,0], '--')
ax0.axvline(x = time_[samples2/2], color="grey")

#ax1.plot(time_, K1)
#ax1.plot(time_, K2)
ax1.axvline(x = time_[sn], color="grey")

ax0.set_ylabel('position [m]')
ax1.set_ylabel('Stiffnes [N/m]')
ax1.set_xlabel('time [s]')
 
plt.savefig('images/stifness_e_th='+str(e_th)+'.png')
plt.show()

''''
plt.figure()
plt.plot(datab[:,0], data_2b_total, label = '2b')
plt.plot(datab[:,0], data_3_long, label = '3')
plt.plot(datab[:,0], data_3_, label = '3 repro')
#plt.plot(time2, data_2b_total)
plt.legend()
plt.show()

'''
'''


if anim == 1:

    data_1_anim = data_1_total
    data_2_anim = data_2_total
    def animate(i):
        x1 = data_1_anim[i]
        x2 = data_2_anim[i]
        line.set_data([x1, x2], [0,0])
        if i > sn:
            line.set_color('g')
    
    def init():
        ax.set_xlim(-L-0.1,L + 0.1)
        ax.set_xlabel('$x$ [m]')
        line.set_data([L/2, -L/2],[0,0])
        return line
    
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set(yticklabels=[])  # remove the tick labels
    
    # These are the objects we need to keep track of.
    line, = ax.plot([], [], 'o-', lw=5, color='#de2d26')
    
    interval = 100*dt
    ani = animation.FuncAnimation(fig, animate, interval=interval, repeat=False, init_func=init)
    
    #f = "images/video.avi" 
    #writergif = animation.PillowWriter(fps=30) 
    #ani.save(f, writer=writergif)
    #ani.save('images/video.mp4', fps = 100)

    plt.show()
'''