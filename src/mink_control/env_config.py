import numpy as np

res = .01
dt = 0.016# time step
# dt = 1/10
t_limit = 5.0# time limit in seconds
thres = (5*np.pi/180)*np.ones(5,dtype=np.float64) # joint error threshold
vel_thres = thres # joint velocity error threshold for stopping
# weighting different parts of the reward function
prox_thres = .05 # proximity threshold - 5 cm
min_prox = 0.1
vel_prox = 0.3

# Controller gains
tau_max = 1.5 #rad/s^2
# damping = .8*tau_max
damping = .5*tau_max
Z = damping 
P = 20
# P = 45
# D = 5
D = P*.8
P_v = 10.0
D_v = 5.0
jnt_vel_max = 1.5 # rad/s
j_max = tau_max/dt  # maximum allowable jerk on the joints
min_H = .14

pi = np.pi
jnt_max = np.array([np.pi, 99*np.pi/180, 99*np.pi/180, np.pi, 103.5*np.pi/180])
jnt_min = np.array([-np.pi, -99*np.pi/180, -99*np.pi/180, -np.pi, -103.5*np.pi/180])

# object settings
obj_radius = .03
obj_label = -1.0
rob_label = 1.0

# S = [S1, S2, S3, S4, S5] offsets
S = np.array([.135+.202, 0, 0, .2675, 0])
# l = [l12, l23, l34, l45, l56] link lengths
l = np.array([0, .2015, 0, 0, .175+.14])
# a = [a12, a23, a34, a45, a56]
a = np.array([np.pi/2, 0, np.pi/2, -np.pi/2, 0])
r = np.sum(S[1:]) + np.sum(l)
limits = np.array([[-r,r],[-r,r],[0,r+S[0]]])
