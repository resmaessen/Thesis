# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:16:32 2022

@author: rosam
"""
from pDMP_functions import pDMP
import numpy as np

def learn(data, phase, K1_2, K1max, DOF, N, alpha, beta, lambd, dt, tau, samples, mode = 1):

    y_old = data[0,0]
    dy_old = data[0,1]
    
    data = []
    data_2_ = []
    
    # create a DMP object
    myDMP = pDMP(DOF, N, alpha, beta, lambd, dt)
    
    
    
    
    # Human to Robot
    for i in range ( samples ):
    
        # generate phase
        phi = phase[i]
        
        # generate an example trajectory (e.g., the movement that is to be learned)
        y = np.array([data[i,0]])
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
    x2_last = data_2_[sn+1,0]
    dx1_last = data_1[sn+1,1]
    
    
    e_th_ = 0.1
    

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