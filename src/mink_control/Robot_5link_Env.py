import numpy as np
from numba import njit,float64, int32, jit
from numba.typed import Dict
from numba.core import types
from mink_control.env_config import *
from mink_control.optimized_functions import calc_jnt_err, PDControl, angle_calc
from mink_control.optimized_functions_5L import nxt_state, proximity, calc_ev_dot, reward_func
# from support_classes import vertex
from mink_control.Object_v2 import rand_object
from mink_control.Robot_5link import forward, reverse, S, l, a, shift, get_coords, calc_eef_vel
from mink_control.sparse_tnsr_replay_buffer import ReplayBuffer

rng = np.random.default_rng()

# @njit((types.Tuple(float64[:],float64[:],float64,int32,types.boolean))
#       (float64[:],float64[:],float64,float64[:],Dict))
@njit(nogil=True)
def env_replay(th, w, t_start, th_goal, obs_dict, steps, S, a, l):
    '''
    inputs:
        th: [th1,th2,th3....]
        w: [w1, w2, ...]
        t_start: time at the start
        th_goal: target joint position to reach
        obs_dict: dictionary containing obs positions
        steps: int num of time steps to simulate
    returns: 
        th: reached th
        w: reached w
        score: the reward of the connections
        t: time step at termination
        flag: collision flag, True if no collsions
    '''
    t = t_start
    jnt_err = calc_jnt_err(th, th_goal)
    dedt  = -1 * w

    score = 0
    flag = True
    done = False
    if t>= t_limit/dt:
        done = True
        flag = False 
        score = -np.inf

    while not done and t<t_start+steps and t < t_limit/dt:
        tau = PDControl(jnt_err, dedt)
        obj_arr = obs_dict[t]
        temp = nxt_state(obj_arr, th, w, tau, a, l, S)
        nxt_th = temp[0:5,0]
        prox = temp[5,0]
        nxt_w = temp[0:5,1]

        t+=1
        jnt_err = calc_jnt_err(nxt_th, th_goal)
        dedt  = -1*w
        th = nxt_th
        w = nxt_w

        if t*dt >= t_limit:
            # print('terminated on t_limit')
            done = True
        elif np.all(np.abs(jnt_err) < np.array([0.03,.03,.03,.03,.03])): # no termination for vel
            # print('terminated by reaching goal')
            done = True
        elif prox < min_prox:
            # print('terminated by collison')
            # flag = False
            done = True

        score += -1 

    return th, w, score, t, flag

def gen_rand_pos(quad, S, l):
    # r = the total reach of the robot
    th = rng.random()*np.pi/2
    phi = np.pi*(rng.random()*20 - 7)/180
    xy = np.array([np.cos(th)*np.cos(phi), np.sin(th)*np.cos(phi), np.sin(phi)])
    r = np.sum(S[1:-1]) + np.sum(l)
    # xy = rng.random(3)
    if quad==2 or quad==3:
        xy[0] = -1*xy[0]
    if quad==3 or quad==4:
        xy[1] = -1*xy[1]
    
    frac = .3
    mag = (r)*.99*frac*rng.random() + r*(1-frac)
    p = mag*xy/np.linalg.norm(xy) + np.array([0,0,S[0]])
    if p[2] < .2:
        p[2] = .2
    if p[2] > r*.5+.2:
        p[2] = r*.5+.2

    # orientation
    keep_trying = True
    while keep_trying:
        u = 2*rng.random(3) - 1
        u = u/np.linalg.norm(u)
        O5_f = p - u*l[-1]
        if O5_f[2] > .2:
            keep_trying = False

    return p, u

def gen_obs_pos(obj_list):
    '''
    obj_list: list of environment objects
    returns: a dictionary whose keys are time steps and items are a
             3xn array with column vectors of the n object locations
    '''
    time_steps = int(np.ceil(t_limit/dt))
    t = 0
    # obs_dict = Dict.empty(key_type=types.int32, 
    #                       value_type=types.float64[:,:])
    obs_dict = Dict.empty(
        key_type=types.int64,
        value_type=types.float64[:,:]
    )
    temp = np.ones((3,len(obj_list)))
    while t <= time_steps:
        i = 0
        for o in obj_list:
            center = o.path(t)
            temp[:,i] = center
            # o.step()
            i+=1
        obs_dict[t] = temp.copy()
        t+=1
    
    return obs_dict

# class action_space():
#     def __init__(self):
#         self.shape = np.array([5]) # three joint angles adjustments
#         self.high = np.ones(5) * tau_max
#         self.low = np.ones(5) * -tau_max

# class observation_space():
#     def __init__(self):
#         self.shape = np.array([5])  

class RobotEnv(object):
    def __init__(self, has_objects=True, num_obj=3, start=None, goal=None, name='robot_5L',batch_size=128,use_goal=False):

        if isinstance(start, np.ndarray):
            self.start = start
            self.goal = goal
        else:
            q1 = rng.choice(np.array([1,2,3,4]))
            q2 = q1 + 2
            if q2 > 4: q2 = q2%4
            # q2 = rng.choice(np.array([1,2,3,4]))
            # while q1 == q2:
            #     q2 = rng.choice(np.array([1,2,3,4]))
            
            s, u = gen_rand_pos(q1, S, l)
            g, v = gen_rand_pos(q2, S, l)

            sol_found = False
            th1 = reverse(s,u,S,a,l)
            th1 = th1[0,:]
            while not sol_found:
                if np.any(np.isnan(th1)): # or np.any(np.abs(th1) >= jnt_max):
                    s,u = gen_rand_pos(q1,S,l)
                    th1 = reverse(s,u,S,a,l)
                    th1 = th1[0,:]
                else:
                    sol_found = True

            sol_found = False
            th2 = reverse(g,v,S,a,l)
            th2 = th2[0,:]
            while not sol_found:
                if np.any(np.isnan(th2)): #or np.any(np.abs(th2) >= jnt_max):
                    g,v = gen_rand_pos(q2,S,l)
                    th2 = reverse(g,v,S,a,l)
                    th2 = th2[0,:]
                else:
                    sol_found = True
        
            self.start = np.clip(th1, jnt_min, jnt_max)
            self.goal = np.clip(th2, jnt_min, jnt_max)

        self.th = self.start
        self.w = np.zeros_like(self.start,dtype=np.float64)
        self.t_step = 0
        self.t_sum = 0
        self.done = False
        max_size = int(np.ceil(t_limit/dt))
        self.memory = ReplayBuffer(max_size,jnt_d=self.th.shape[0], time_d=6, file=name)
        self.batch_size = batch_size
        self.info = {}
        self.jnt_err = calc_jnt_err(self.th, self.goal)
        self.dedt = np.zeros_like(self.jnt_err)
        weights = rng.random(3)
        weights = np.round(weights/np.linalg.norm(weights),2)
        self.weights = weights
        self.info["converged"] = False
        self.use_goal = use_goal

        if has_objects:
            objs = []
            i = 0
            k = 0
            q_list = []
            while i < (num_obj) and k < int(1e3):
                k+=1
                quad = int(rng.choice([1,2,3,4]))
                if len(q_list)>0 and len(q_list)<4:
                    while quad in q_list:
                        quad = int(rng.choice([1,2,3,4]))
                o = rand_object(q=quad)
                prox = np.inf 
                can_use = True
                mx_t = int(np.ceil(o.tf/dt))
                for j in range(mx_t):
                    pos_j = o.path(j)
                    _,prox_s = proximity(pos_j, self.start, a, l,S)
                    _,prox_g = proximity(pos_j, self.goal, a,l,S)
                    if prox_s < vel_prox or prox_g <= min_prox:
                        j = mx_t+1
                        can_use = False
                if can_use:
                    objs.append(o)
                    q_list.append(quad)
                    i += 1
            self.objs = objs
        else:
            self.objs = []

    # def env_replay(self, start_v:vertex, th_goal, obs_dict, steps):
    #     if not isinstance(th_goal, np.ndarray):
    #         th_goal = np.array(th_goal,dtype=np.float64)
    #     th = np.array(start_v.th,dtype=np.float64)
    #     w = start_v.w
    #     w = w.astype(np.float64)
    #     t_start = start_v.t
    #     t_start = int(t_start)
    #     # th, w, score, t, flag = env_replay(th,w,t_start, th_goal, obs_dict, steps)
    #     # self.th = th
    #     # self.w = w
    #     # self.jnt_err = calc_jnt_err(th, self.goal)
    #     # self.dedt = -1*w
    #     return env_replay(th,w,t_start, th_goal, obs_dict, steps,S,a,l)

    def reward(self, eef_vel, prox):
        return -1
    
    def step(self, action, use_PID=False, eval=False):  
        objs_arr = np.zeros((3,len(self.objs)))
        nxt_objs_arr = np.zeros_like(objs_arr)
        for i in range(len(self.objs)):
            o = self.objs[i]
            objs_arr[:,i] = o.path(self.t_step)
            nxt_objs_arr[:,i] = o.path(self.t_step+1)
        
        if use_PID:
            err = self.jnt_err
            dedt = self.dedt
            action = PDControl(err, dedt)
            self.info['action'] = action
            package = nxt_state(objs_arr, self.th, self.w, action, a, l, S)
        else:
            if not isinstance(action,np.ndarray):
                action = action.detach().cpu().numpy()
                action = action.astype(np.float64)
                action = action.reshape(5)
            package = nxt_state(objs_arr, self.th, self.w, action, a, l, S)
        
        nxt_th = package[0:self.th.shape[0],0]
        nxt_w = package[0:self.w.shape[0],1]
        prox = package[-1,0]

        # need to add a eef vel function
        eef_vel,eef = calc_eef_vel(nxt_th,nxt_w)
        v_dot = calc_ev_dot(eef,eef_vel,nxt_objs_arr)
        reward = reward_func(prox, v_dot, self.weights)
        
        self.t_step += 1
        err = calc_jnt_err(nxt_th, self.goal)
        self.jnt_err = err
        if self.t_step*dt >= t_limit:
            done = True
        elif np.all(abs(err) < thres) and np.all(abs(nxt_w) < vel_thres):
            done = True
            self.info["converged"] = True 
            reward += 10
        else:
            done = False

        self.th = nxt_th
        self.w = nxt_w
        self.dedt = -1*self.w
        coords,feats = [],[]
        if not eval:
            rob_coords, rob_feats = get_coords(nxt_th, self.t_step)
            for o in self.objs:
                c,f = o.get_coords(self.t_step)
                coords.append(c)
                feats.append(f)
            coords.append(rob_coords)
            feats.append(rob_feats)
            coords = np.vstack(coords)
            feats = np.vstack(feats)
        
        state = (coords, feats, self.jnt_err, self.dedt)
        return state, reward, done, self.info
    
    def get_state(self):
        coords,feats=[],[]
        rob_coords,rob_feats = get_coords(self.th, self.t_step)
        for o in self.objs:
            c,f = o.get_coords(self.t_step)
            coords.append(c)
            feats.append(f)
        coords.append(rob_coords)
        feats.append(rob_feats)
        coords = np.vstack(coords)
        feats = np.vstack(feats)

        state = (coords, feats, self.jnt_err, self.dedt)
        return state 

    def reset(self):
        self.th = self.start
        self.w = np.zeros_like(self.th, dtype=float)
        self.jnt_err = calc_jnt_err(self.th, self.goal)
        self.t_step = 0
        self.dedt = -1*self.w
        self.memory.clear()
    
    def copy(self):
        env = RobotEnv(start=self.start, goal=self.goal)
        env.objs = []
        env.objs = self.objs
        env.weights = self.weights
        env.jnt_err = self.jnt_err
        return env

    def store_transition(self,state, action, reward, new_state,done,t_step):
        self.memory.store_transition(state, self.weights, action, reward,new_state,done,t_step)

    def sample_memory(self):
        return self.memory.sample_buffer(self.batch_size)

