# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 2022

@author: Rosa Maessen

This code is based on the AFS_structure code from Tadej Petric
https://gitlab.com/tpetric/AFS_C/-/blob/master/AFSstructure.cpp
Instead of C++, Python is used!

following this paper: https://journals.sagepub.com/doi/pdf/10.1177/0278364911421511


"""

import numpy as np

class AFS:
    
    def __init__(self, DOF_, N_, ni_, K_): 
        # Adaptive oscillar parameters
        self.flag = 0
        self.DOF = DOF_
        self.N = N_
        
        self.ni = ni_
        self.K = K_
        
        # Current state for integration
        self.phi = np.zeros([self.DOF])
        self.Omega = np.zeros([self.DOF])
        
        # Signals
        self.y_in = np.zeros([self.DOF])   # Input Signal
        self.y_fb = np.zeros([self.DOF])   # Aproximated signal (adaptive fourier signal)
        
        # Current state for integration of adaptive fourier series coeficients
        self.alpha = np.zeros([self.DOF,self.N])
        self.beta = np.zeros([self.DOF,self.N])
        
    
    
    def get_ni(self):
        return self.ni
    
    def get_K(self):
        return self.K

    def get_dof(self):
        return self.dof

    def get_N(self):
        return self.N
    
    def get_flag(self):
        return self.flag
        
    def get_input(self,i):
        return self.y_in[i]
    
    def get_output(self,i):
        return self.y_fb[i]
    
    def update_input(self, i, y_in_):
        self.y_in[i] = y_in_
    
    
    def get_phase(self, i):
        return self.phi[i]
    
    def get_frequency(self, i):
        return self.Omega[i] 
    
    def set_frequency(self, i, Omega_):
        self.Omega[i] = Omega_
        
    def set_flag(self, flag_):
        self.flag = flag_
    
    
        
    def set_initial_AFS_state(self, frequency):
        
        for j in range(self.DOF):
            self.y_fb[j] = 0    # Initial state before first calculation
            
            self.phi[j] = 0
            self.Omega[j] = frequency
            
            for i in range(self.N):
                self.alpha[j,i] = 0
                self.beta[j,i] = 0
    
    
  

    def AFS_param_print(self):
        print("\ndof = %d, N = %d, flag = %d\n",self. dof, self.N, self.flag);
        print("ni = %.2lf, K = %.2lf\n", self.ni, self.K)
        
        
        print("phi = (");
        for j in range(self.DOF-1):
            print("%.2lf, ", self.phi[j]);
            print("%.2lf)\n\n", self.phi[self.DOF-1])

     
        print("Omega = (")
        for j in range(self.DOF-1):
            print("%.2lf, ", self.Omega[j]);
            print("%.2lf)\n\n", self.Omega[self.DOF-1])
       
        for j in range(self.DOF):
            print("alpha = (");
            for i in range(self.N-1):
                print("%.2lf, ", self.alpha[j,i])
            
            print("%.2lf)\n\n", self.alpha[j,i])

        for j in range(self.DOF):
            print("beta = (");
            for i in range(self.N-1):
                print("%.2lf, ", self.beta[j,i])
            print("%.2lf)\n\n", self.beta[j,i])

        
    
    def AFS_integrate(self, dt, steps):
        
        #servo_rate = 1                  # Servo_base_rate / Task_servo_ratio
        #desired_dt = 1.0/servo_rate     # Servo_rate
        
        #steps = round(dt / desired_dt + 0.5)
        #steps = 10
        dt = dt/steps
        
        self.y_fb = np.zeros([self.DOF]) 
   
        
        for k in range(steps):
            for j in range(self.DOF):
                
                if self.flag == 1: # check if learning is active !
                    e = self.y_in[j] - self.y_fb[j] # feedback error
                else: 
                    e = 0
                
                self.y_fb[j] = 0 # set to 0 for recalculation in the next loop
                
                for i in range(self.N):
                    dalpha = e * self.ni * np.cos(i*self.phi[j])
                    
                    self.alpha[j,i] += dalpha * dt
                    
                    
                    dbeta = e * self.ni * np.sin(i*self.phi[j])
                    self.beta[j,i] += dbeta * dt
                    
                    self.beta[j][1] = 0;  # this is requred otherwice the phase is not clearly defined!!!
                    
                    self.y_fb[j] += ( ( self.alpha[j,i] * np.cos(i*self.phi[j])) + ( self.beta[j,i] * np.sin(i* self.phi[j])) )
                    
                dphi = self.Omega[j] - (self.K * e * np.sin(self.phi[j]))
    
                dOmega = - self.K * e * np.sin(self.phi[j])

                #   Euler integration - new frequency and phase
                self.phi[j] += dphi * dt
                self.Omega[j] += dOmega * dt
                


