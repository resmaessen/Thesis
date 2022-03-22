# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 2022

@author: Rosa Maessen

This code is based on the PDMP_structure code from Tadej Petric
https://gitlab.com/tpetric/pDMP_C/-/blob/master/pDMPstructure.cpp
Instead of C++, Python is used!

"""

import math
import numpy as np



class pDMP_structure:
    def __init__(self, alpha_z, beta_z, c, dof, f, g, lambda_, N, P, phi, r, tau, w):
        
        self.dof = dof
        
        # pDMP - periodicDMP parameters        
        self.alpha_z, self.beta_z = alpha_z, beta_z #  DMP gains
        self.N = N              # Number of basis functions
        self.lambda_ = lambda_  # Forgetting factor
        self.w = w              # Weights
        self.P = P 
        self.c = c              # Basis functions
        self.f = f              # Nonlinear turn
        self.r = r              # Amplitude parameter
               
        # Goal (anchor of oscilation)               
        self.g = g
        
        
        # Phase and frequency scaling
        self.phi = phi
        self.tau = tau
        

    def pDMP_calculate_f(self, y_in, dy_in, ddy_in):
        
        alpha_z, beta_z, c, dof, f, g, lambda_, N, P, phi, r, tau, w = \
            self.alpha_z, self.beta_z, self.c, self.dof, self.f, self.g, self.lambda_, self.N, self.P, self.phi, self.r, self.tau, self.w
        
        target = []
        psi = []
        
        for i in range(dof):
            # Calculate target for fitting
            target = tau[i]**2 * ddy_in[i] - alpha_z*(beta_z*(g[i]*y_in[i])-tau[i]*dy_in[i])
            
            sum_all  = 0
            sum_psi = 0
            
            # Recursie regression
            for j in range(N):
                # Update weights
                psi[j] = math.exp(2.5*N* (math.cos(phi[i]-c[j])-1))
                Pu = P[i,j]
                Pu = (Pu - (Pu**2 * r[i]**2)/ (lambda_/psi[j] + r[i]* r[i]*Pu))/lambda_
                w[i,j] += psi[j]*Pu*r[i]*(target[i]-w[i,j]*r[i])
                P[i,j] = Pu
                # Reconstructe DMP
                sum_psi += psi[j]
                sum_all += w[i,j]*r[i]*psi[j]
            
            if sum_psi == 0:
                f[i] = 0
            else:
                f[i] = sum_all/sum_psi
                
        
