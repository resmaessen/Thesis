# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 10:38:53 2022

@author: Rosa Maessen
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pDMP_functions import pDMP


plt.close('all')

anim = 0

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

K2max = 100
K2 = []

for i in range(samples):
    if np.sin(phase[i])>=0:
        K2.append([K2max])
    else:
        K2.append([0])

plt.figure()
plt.plot(K2)