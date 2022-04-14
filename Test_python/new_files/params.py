# EXPERIMENT PARAMETERS
dt = 0.05 # system sample time
exp_time = 40 # total experiment time
samples = int(1/dt) * exp_time

DOF = 1 # degrees of freedom (number of DMPs to be learned)
N = 50 # number of weights per DMP (more weights can reproduce more complicated shapes)
alpha = 8 # DMP gain alpha
beta = 2 # DMP gain beta
lambd = 0.995 # forgetting factor
tau = 1 # DMP time period = 1/frequency (NOTE: this is the frequency of a periodic DMP)

mode = 1 # DMP mode of operation (see below for details)


L = 0.15  # Length movement

K1max = 1100 # N/m
K2max = 1100 # N/m

c1 = 110 # Damping coeff
c2 = 110 # Damping coeff

m = 10 # kg
g = 9.81 # Gravitation

mu_s = 0.2
mu_k = 0.15
mu_s_random = 0
mu_k_random = 0

h = 2.5


e_th = 0.02

F_y = 5
F_n = m*g + F_y

F_fs_max = mu_s * N
F_fk = mu_k*N
