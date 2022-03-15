# EXPERIMENT PARAMETERS
dt = 0.05 # system sample time
exp_time = 40 # total experiment time
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

c1 = 120
c2 = 120

m = 10 # kg
g = 9.81

mu_s = 0.3
mu_k = 0.2
mu_s_random = 0
mu_k_random = 0


e_th = 0.02

ni = 2
K = 20
M = 10
AFS_step = 30

F_y = 5
F_n = m*g + F_y

F_fs_max = mu_s * N
F_fk = mu_k*N