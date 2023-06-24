from numba import njit, jit, int32, float64
import numpy as np
# from env_config import P, D, tau_max
P = 40
D = .8*40
tau_max = 1.5

@njit(nogil=True)
def clip(arr, limits):
    for i in range(arr.shape[0]):
        if abs(arr[i]) > limits[i]:
            if arr[i] < 0:
                arr[i] = -1*limits[i]
            else:
                arr[i] = limits[i]
    return arr 

@njit((float64[:,:])(float64, float64, float64, float64), nogil=True)
def T_ji(thj, aij, lij, Sj):
    return np.array([[np.cos(thj),            -np.sin(thj),              0,            lij],
                     [np.sin(thj)*np.cos(aij), np.cos(thj)*np.cos(aij), -np.sin(aij), -np.sin(aij)*Sj],
                     [np.sin(thj)*np.sin(aij), np.cos(thj)*np.sin(aij),  np.cos(aij),  np.cos(aij)*Sj],
                     [                    0.0,                     0.0,          0.0,             1.0]])

@njit((float64[:,:])(float64,float64),nogil=True)
def T_1F(ph, Sj):
    return np.array([[np.cos(ph), -np.sin(ph), 0.0, 0.0],
                    [np.sin(ph), np.cos(ph), 0.0, 0.0],
                    [0.0, 0.0, 1.0, Sj],
                    [0.0, 0.0, 0.0, 1.0]])

@njit((float64[:,:])(float64[:,:]),nogil=True)
def T_inverse(T):
    R = T[0:3,0:3].T
#     print(R)
    temp = T[0:3,3]
#     print('shift ' + str(shift))
    translation = -1*(R@temp)
#     print('-shift ' + str(shift))
    new_T = np.zeros((4,4))
    for i in range(0,3):
        for j in range(0,3):
            new_T[i,j] = R[i,j]
    for i in range(0,3):
        new_T[i,3] = translation[i]
    new_T[3,3] = 1
    return new_T


@njit((float64[:])(float64[:],float64[:],float64[:],float64[:],float64[:]),nogil=True)
def asb_link2(obj_pos, th, a, l, S):
    T_2toF = T_1F(th[0], S[0])@T_ji(th[1],a[0],l[0],S[1])
    inv_T = T_inverse(T_2toF)
    vec = np.array([obj_pos[0],obj_pos[1],obj_pos[2],1])
    return inv_T@vec

@njit((float64[:])(float64[:],float64[:],float64[:],float64[:],float64[:]),nogil=True)
def asb_link3(obj_pos, th, a, l, S):
    T_3toF = T_1F(th[0], S[0])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[2],a[1],l[1],S[2])
    inv_T = T_inverse(T_3toF)
    vec = np.array([obj_pos[0],obj_pos[1],obj_pos[2],1])
    return inv_T@vec

@njit((float64)(float64[:],float64[:],float64[:],float64[:],float64[:]))
def proximity(obj_pos, th, a, l, S):
    th = th + np.array([0,np.pi/2,0])
    asb_l2 = asb_link2(obj_pos,th,a,l,S)
    asb_l3 = asb_link3(obj_pos,th,a,l,S)
    prox1 = np.inf
    if asb_l2[0] <= 0.0:
        prox1 = np.linalg.norm(asb_l2[0:3])
    elif asb_l2[0] >= l[1]:
        prox1 = np.linalg.norm(asb_l2[0:3] - np.array([l[1],0.0,0.0]))
    else:
        prox1 = np.linalg.norm(asb_l2[1:3])
    
    prox2 = np.inf
    if asb_l3[0] <= 0.0:
        prox2 = np.linalg.norm(asb_l3[0:3])
    elif asb_l3[0] >= l[2]:
        prox2 = np.linalg.norm(asb_l3[0:3] - np.array([l[2],0.0,0.0]))
    else:
        prox2 = np.linalg.norm(asb_l3[1:3])

    if prox1 < prox2:
        return prox1
    else:
        return prox2

@njit((float64[:])(float64[:],float64[:],float64[:],float64[:],float64[:]),nogil=True)
def njit_forward(th, a, l, S, P_3):
    th = th + np.array([0,np.pi/2,0])
    return T_1F(th[0],S[0])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[2],a[1],l[1],S[2])@P_3    

@njit((float64[:])(float64[:],float64[:]),nogil=True)
def calc_jnt_err(curr,goal):
    # first two 
    s1 = np.sin(curr)
    c1 = np.cos(curr)
    curr = np.arctan2(s1,c1)
    s2 = np.sin(goal)
    c2 = np.cos(goal)
    goal = np.arctan2(s2,c2)

    jnt_err = goal - curr

    return jnt_err

@njit((float64[:])(float64[:]))
def angle_calc(th):
    s = np.sin(th)
    c = np.cos(th)
    arr = np.arctan2(s,c)
    return arr

@njit((float64[:])(float64[:],float64[:]),nogil=True)
def PDControl(jnt_err, dedt):
    tau = P*jnt_err + D*dedt
    max_vec = np.ones(5,dtype=np.float64)*tau_max
    tau = clip(tau,max_vec)
    return tau
