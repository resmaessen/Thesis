import numpy as np
import matplotlib.pyplot as plt
from pDMP_functions import pDMP
from AFS_structure import AFS
from matplotlib import gridspec 

plt.close('all')


# EXPERIMENT PARAMETERS
dt = 0.01 # system sample time
exp_time = 60 # total experiment time
#samples = int(1/dt) * exp_time

DOF = 1 # degrees of freedom (number of DMPs to be learned)
N = 25 # number of weights per DMP (more weights can reproduce more complicated shapes)

ni = 2
K = 20
M = 10


t = np.arange(0, exp_time, dt)
samples = len(t)

y = np.zeros(len(t))

samples = len(t)
omega_t = []


## Trajectory 
for idx, t_ in enumerate(t):
    if t_ < 10:
        w = 4*np.pi        
    elif t_<20:
        w = (2*np.pi)
    elif t_<45:
        w = (5*np.pi)
    else:
        w = (3*np.pi)

    y[idx] = np.sin(w*t_)
    omega_t.append(w)
    
omega_t = np.asarray(omega_t)


## Setup AFS
myAFS = AFS(DOF, M, ni, K)
myAFS.set_flag(1)


frequency_last = np.pi

myAFS.set_initial_AFS_state(frequency_last)

data = []
phase_last = 0

for i in range(samples):
        
   
    myAFS.update_input(0, np.array([y[i]]))

    myAFS.AFS_integrate(dt)        
    
    output = myAFS.get_output(0)
    phase_last = myAFS.get_phase(0) 
    frequency_last = myAFS.get_frequency(0)
    
    
    data.append([output, phase_last, frequency_last, (omega_t[i] - (phase_last))**2])
        
    
data = np.asarray(data) 

### Plot results
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)


ax0.plot(t, y)
ax0.plot(t, data[:,0])
ax1.plot(t, omega_t, '--', label = r'$\omega_t$')
ax1.plot(t, data[:,2], label = r'$\Omega$')

ax1.legend(loc = 'upper right')

ax0.set_xticklabels([])
ax1.set_xticklabels([])


ax0.set_ylabel('y', fontsize='13')
ax1.set_ylabel(r'$\Omega \ [rad \ s^{-1}]$', fontsize='13')
ax1.set_xlabel('time [s]', fontsize='12')

plt.show()