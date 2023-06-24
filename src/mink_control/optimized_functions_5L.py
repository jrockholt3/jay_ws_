from numba import njit, int32, float64
import numba as nb
from mink_control.optimized_functions import T_1F, T_ji, T_inverse, calc_jnt_err, angle_calc, PDControl, clip
import numpy as np
from mink_control.env_config import *
from mink_control.env_config import damping as Z
from mink_control.Robot_5link import shift, get_transforms, points

def create_store_state(env):
    objs_arr = np.zeros((3,len(env.objs)))
    for i in range(len(env.objs)):
        o = env.objs[i]
        objs_arr[:,i] = o.path(env.t_step)
    
    return (env.th, objs_arr, env.jnt_err, env.dedt)

@njit((float64[:,:])(float64), nogil=True)
def Rot_Z(th):
    c = np.cos(th)
    s = np.sin(th)
    T = np.array([[  c,  -s, 0.0, 0.0],
                  [  s,   c, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    return T.T

# nb.types.UniTuple(float64,2)
@njit(nb.types.UniTuple(float64,2)(float64[:],float64[:],float64[:],float64[:],float64[:]), nogil=True)
def proximity(obj_pos, th, a, l, S):
    '''
    input 
        obj_pos = [x,y,z] of obj
    '''
    step_size = .05 # step size along robot arm in meters
    point_arr = points(th,S,a,l)
    O2 = point_arr[0:3,1]
    O3 = point_arr[0:3,2]
    O5 = point_arr[0:3,3]
    Ptool_f = point_arr[0:3,4] 
    u23 = np.linalg.norm(O3-O2)
    u23_f = (O3 - O2)/u23
    u35 = np.linalg.norm(O5-O3)
    u35_f = (O5 - O3)/u35
    u5t = np.linalg.norm(Ptool_f-O5)
    u5t_f = (Ptool_f - O5)/u5t
    max_size = int(np.ceil((u23+u35+u5t)/step_size))
    p_min = 1*np.inf
    r_min = 1*np.inf
    s = 0.0
    for i in range(max_size):
        if s < u23:
            u_i = s*u23_f + O2
            r_i = np.inf
        elif s < u23 + u35:
            u_i = u35_f * (s - u23) + O3
            r_i = np.linalg.norm(u_i)
        elif s <= u23 + u35 + u5t:
            u_i = u5t_f * (s - u23-u35) + O5
            r_i = np.linalg.norm(u_i)
            
        prox_i = np.linalg.norm(obj_pos - u_i)
        if prox_i < p_min:
            p_min = prox_i
        if r_i < r_min:
            r_min = r_i
        s += step_size
    z_min = min([O3[2],O5[2],Ptool_f[2]])
    r_prox = r_min - .28 + min_prox 
    env_prox = min([p_min, z_min, r_prox])
    return env_prox, p_min


@njit(nogil=True)
def calc_ev_dot(eef,eef_vel,obj_arr):
    r,c = obj_arr.shape
    v_dot_sum = 0
    for i in range(c):
        temp = obj_arr[:,c] - eef
        temp = temp/np.linalg.norm(temp)
        v_dot = np.dot(eef_vel,temp)
        if v_dot > 0:
            v_dot_sum += v_dot
    return v_dot_sum/c 

@njit(nogil=True)
def reward_func(prox, ev_dot, w):
    return -1*w[0] + w[1]*(np.exp(-prox**2/0.5**2) - 1) + w[2]*(np.exp(-ev_dot**2) - 1)

@njit(nogil=True) # (float64[:])(float64,float64),
def calc_clip_vel(prox):
    prox_clip = np.ones(5) * jnt_vel_max * (1 - np.exp(-(prox-min_prox)**2/(.659*(vel_prox-min_prox))**2))
    z_clip = np.ones(5) * jnt_vel_max #* (1 - np.exp(-(zmin-.14)**2/(1.61*(zmin-.34))**2))
    z_clip[0] = jnt_vel_max
    clip_vec = np.ones(5)
    for i in range(5):
        clip_vec[i] = min(z_clip[i], prox_clip[i])
    # print('prox',prox,'prox_clip',prox_clip)
    return clip_vec

# @njit((float64[:,:])(float64[:,:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]),nogil=True)
# @njit(nogil=True)
def nxt_state(obj_pos, th, w, tau, a, l, S):
    '''
    obj_pos = [pos1, pos2,...]
    '''
    prox_arr = np.zeros(obj_pos.shape[1])
    obj_prox_arr = np.zeros(obj_pos.shape[1])
    for i in range(0, obj_pos.shape[1]):
        prox_arr[i],obj_prox_arr[i] = proximity(obj_pos[:,i], th, a, l, S)
    prox = np.min(prox_arr)
    obj_prox = np.min(obj_prox_arr)

    vel_clip = calc_clip_vel(prox)
    # print('prox',prox,'vel_clip',vel_clip)

    if prox <= min_prox:
        paused = True
    else:
        paused = False

    if not paused:
        nxt_w = (tau - Z*w)*dt + w
        if np.any(np.abs(nxt_w) > vel_clip):
            nxt_w = clip(nxt_w, vel_clip)
            tau = (nxt_w - w)/dt + Z*w
        
        nxt_th = (tau - Z*w)*dt**2/2 + w*dt + th
    else:
        nxt_w = np.zeros(w.shape[0])
        nxt_th = th

    if np.any(nxt_th >= jnt_max) or np.any(nxt_th <= jnt_min):
        nxt_w[nxt_th >= jnt_max] = 0
        nxt_w[nxt_th <= jnt_min] = 0
        nxt_th[nxt_th >= jnt_max] = jnt_max[nxt_th >= jnt_max]
        nxt_th[nxt_th <= jnt_min] = jnt_min[nxt_th <= jnt_min]

    package = np.zeros((6,2))
    for i in range(0,nxt_th.shape[0]):
        package[i,0] = nxt_th[i]

    for i in range(0,nxt_w.shape[0]):
        package[i,1] = nxt_w[i]

    package[-1,0] = obj_prox
    return package
