"""
PERIODIC DYNAMIC MOVEMENT PRIMITIVES (pDMP)

This periodic DMP system has 3 modes:
    - LEARN: learns the DMP shape based on some input signal
    - UPDATE: updates the DMP shape based on some input signal
    - REPEAT: repeats the existing DMP


AUTHOR: Luka Peternel
e-mail: l.peternel@tudelft.nl


REFERENCE:
L. Peternel, T. Noda, T. Petrič, A. Ude, J. Morimoto and J. Babič
Adaptive control of exoskeleton robots for periodic assistive behaviours based on EMG feedback minimisation,
PLOS One 11(2): e0148942, Feb 2016

"""

import numpy as np


class pDMP:
    # INITIALISATION
    def __init__(self, DOF, N, alpha, beta, lambd, dt, h = 2.5):

        # settings
        self.DOF = DOF # degrees of freedom (number of DMPs)
        self.N = N # number of kernel functions per DMP
        self.alpha = alpha # DMP gain alpha
        self.beta = beta # DMP gain beta
        self.lambd = lambd # forgetting factor
        self.dt = dt # sample time
        self.h = h # width of Gaussian

        # DMP learning variables
        self.f = np.zeros([self.DOF]) # shape function
        self.w = np.zeros([self.DOF,self.N]) # DMP weights
        self.c = np.zeros([self.N]) # centers of kernel functions
        self.P = np.ones([self.DOF,self.N]) # regression variable P
        self.r = np.ones([self.DOF]) # amplitude parameter
        self.g = np.zeros([self.DOF]) # DMP goal variable

        # DMP state variables
        self.y = np.zeros([self.DOF])
        self.z = np.zeros([self.DOF])

        # DMP phase and period
        self.phi = np.zeros([self.DOF])
        self.tau = np.zeros([self.DOF])

        # define centers of kernel functions
        spread = ( 2 * np.pi - 0.0 ) / N # distance between the kernels
        for i in range(self.N):
            self.c[i] = 0.5 * spread + spread * i




    # GET STATE
    def get_state(self):
        dy = self.z / self.tau
        return self.y, dy, self.tau, self.phi


    def set_state(self, y):
        self.y = y
        #self.z = z

        if self.y.shape != np.zeros([self.DOF]).shape:
            raise NameError('Shape y is not correct')
        #if self.z.shape != np.zeros([self.DOF]).shape:
        #    raise NameError('Shape z is not correct')

    # SET DMP PHASE
    def set_phase(self, phi):
        self.phi = phi


    def set_dt(self, dt):
        self.dt = dt

    # SET DMP PERIOD
    def set_period(self, tau):
        self.tau = tau




    # SET DMP WEIGHTS
    def set_weights(self, DOF, w):
        self.w[DOF,:] = w




    # GET DMP WEIGHTS
    def get_weights(self, DOF):
        return self.w[DOF,:]




    # SET DMP KERNELS
    def set_kernels(self, c):
        self.c = c




    # GET DMP KERNELS
    def get_kernels(self):
        return self.c




    # LEARN MODE
    def learn(self, y, dy, ddy):
        f_d = np.zeros([self.DOF])
        psi = np.zeros([self.N])

        for i in range(self.DOF):
            psi_sum = 0
            weighted_sum = 0

            # desired shape

            f_d[i] = self.tau[i]**2 * ddy[i] - self.alpha * (self.beta * ( self.g[i] - y[i] ) - self.tau[i] * dy[i])

            # recursive least-squares regression
            for j in range(self.N):
                # update kernels and weights
                psi[j] = np.exp( self.h * self.N * ( np.cos( self.phi[i]- self.c[j] ) - 1 ) )
                P_new = ( self.P[i,j] - ( self.P[i,j]**2 * self.r[i]**2 ) / ( self.lambd / psi[j] + self.P[i,j] * self.r[i]**2 ) ) / self.lambd
                self.w[i,j] += psi[j] * P_new * self.r[i] * ( f_d[i] - self.w[i,j] * self.r[i] )
                self.P[i,j] = P_new

                # sum kernels and weights
                weighted_sum += self.w[i,j] * psi[j] * self.r[i]
                psi_sum += psi[j]

            # make sure there is no division with zero
            if ( psi_sum == 0 ):
                self.f[i] = 0
            else:
                self.f[i] = weighted_sum / psi_sum




    # UPDATE MODE
    def update(self, U):
        psi = np.zeros([self.N])

        for i in range(self.DOF):
            psi_sum = 0
            weighted_sum = 0

            # recursive least-squares regression
            for j in range(self.N):
                # update kernels and weights
                psi[j] = np.exp( self.h * self.N * ( np.cos( self.phi[i] - self.c[j] ) - 1 ) )
                P_new = ( self.P[i,j] - ( self.P[i,j]**2 * self.r[i]**2 ) / ( self.lambd / psi[j] + self.P[i,j] * self.r[i]**2 ) ) / self.lambd
                self.w[i,j] += psi[j] * P_new * self.r[i] * U[i]
                self.P[i,j] = P_new

                # sum kernels and weights
                weighted_sum += self.w[i,j] * psi[j] * self.r[i]
                psi_sum += psi[j]

            # make sure there is no division with zero
            if ( psi_sum == 0 ):
                self.f[i] = 0
            else:
                self.f[i] = weighted_sum / psi_sum




    # REPEAT MODE
    def repeat(self):
        psi = np.zeros([self.N])

        for i in range(self.DOF):
            psi_sum = 0
            weighted_sum = 0

            # recursive least-squares regression
            for j in range(self.N):
                psi[j] = np.exp( self.h * self.N * ( np.cos(self.phi[i] - self.c[j] ) - 1 ) )

                # sum kernels and weights
                weighted_sum += self.w[i,j] * psi[j] * self.r[i]
                psi_sum += psi[j]

            # make sure there is no division with zero
            if ( psi_sum == 0 ):
                self.f[i] = 0
            else:
                self.f[i] = weighted_sum / psi_sum




    # INTEGRATION
    def integration(self):
        for i in range(self.DOF):
            dz = ( 1 / self.tau[i] ) * ( self.alpha * ( self.beta * ( self.g[i] - self.y[i] ) - self.z[i] ) + self.f[i] )
            dy = ( 1 / self.tau[i] ) * self.z[i]

            self.y[i] += dy * self.dt
            self.z[i] += dz * self.dt
