import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pDMP_functions import pDMP
from AFS_structure import AFS
from matplotlib import gridspec 
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
        
    #if abs(F) < mu_s_ * F_n:
    #    F = 0
    #else:
    #F = F #- mu_k_*N*np.sign(F) 
        
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


## Setup AFS
myAFS = AFS(DOF, M, ni, K)
myAFS.set_flag(1)


frequency_last = 2*np.pi

myAFS.set_initial_AFS_state(frequency_last)

data = []
phase_last = 0

for i in range(samples):
        
    myAFS.update_input(0, np.array([data_2[i,0]]))

    myAFS.AFS_integrate(dt)        
    
    output = myAFS.get_output(0)
    phase_last = myAFS.get_phase(0) 
    frequency_last = myAFS.get_frequency(0)
    
    
    data.append([output, phase_last, frequency_last])
        
    
data = np.asarray(data) 

### Plot results
fig = plt.figure()
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1,1]) 

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)
ax2 = plt.subplot(gs[2], sharex = ax0)


ax0.plot(data_2[:,0])
ax1.axhline(y= 2*np.pi)
ax1.plot(data[:,2], '--', label = r'$\omega$')

ax2.plot(data[:,1])

ax1.legend(loc = 'upper right')

ax0.set_xticklabels([])
ax1.set_xticklabels([])


ax0.set_ylabel('y', fontsize='13')
ax1.set_ylabel(r'$\Omega \ [rad \ s^{-1}]$', fontsize='13')
ax1.set_xlabel('time [s]', fontsize='12')

plt.show()



'''
### Plot results
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)


ax0.plot(t, y)
ax1.plot(t, omega_t, '--', label = r'$\omega_t$')
ax1.plot(t, data[:,2], label = r'$\Omega$')

ax1.legend(loc = 'upper right')

ax0.set_xticklabels([])
ax1.set_xticklabels([])


ax0.set_ylabel('y', fontsize='13')
ax1.set_ylabel(r'$\Omega \ [rad \ s^{-1}]$', fontsize='13')
ax1.set_xlabel('time [s]', fontsize='12')

plt.show()
'''