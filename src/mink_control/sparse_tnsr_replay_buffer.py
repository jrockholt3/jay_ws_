import numpy as np
import pickle 
from mink_control.Object_v2 import obj_get_coords
from mink_control.Robot_5link import get_coords
# import torch
# import gc 

rng = np.random.default_rng()

class ReplayBuffer:
    def __init__(self, max_size=int(1e6), jnt_d=5, time_d=6, file='replay_buffer_no_obj',dir='buffer'):
        # jnt_d = joint dimensions
        self.mem_size = max_size
        self.mem_cntr = 0
        self.jnt_d = jnt_d
        self.time_d = time_d # this is the # of time steps that will be looked over
        # coord_memory, feat_memory, and jnt_err all define the state 
        self.th_memory = np.zeros((max_size,jnt_d),dtype=np.float32)
        self.obj_memory = np.empty(max_size,dtype=np.object)
        self.jnt_err_memory = np.zeros((self.mem_size, jnt_d),dtype=np.float32)
        self.jnt_dedt_memory = np.zeros((self.mem_size, jnt_d),dtype=np.float32)

        self.new_th_memory = np.zeros((max_size,jnt_d),dtype=np.float32)
        self.new_obj_memory = np.empty(max_size,dtype=np.object)
        self.new_jnt_err_memory = np.zeros((self.mem_size, jnt_d),dtype=np.float32)
        self.new_jnt_dedt_memory = np.zeros((self.mem_size, jnt_d), dtype=np.float32)
        self.weight_memory = np.zeros((self.mem_size,3),dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, jnt_d),dtype=np.float64)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.file = dir + '/' + file
        self.time_step = np.ones(self.mem_size)*np.inf # variable for sampling buffer
                                        # will need to go back 'x' # of time steps
                                        # to create the 4D space tensor
                                        # set to np.inf so that all entries do not 
                                        # have a state stored will have a greater timestep
                                        # and avoid storing empty arrays

    def store_transition(self, state, weight, action, reward, new_state, done, time_step):

        # state is (coord_list, feat_list, jnt_err, jnt_dedt)
        ndx = self.mem_cntr % self.mem_size
        self.th_memory[ndx] = state[0]
        self.obj_memory[ndx] = state[1]
        self.jnt_err_memory[ndx] = state[2]
        self.jnt_dedt_memory[ndx] = state[3]
        self.weight_memory[ndx] = weight
        if isinstance(action, np.ndarray):
            self.action_memory[ndx] = action
        else:
            self.action_memory[ndx] = action.detach().cpu().numpy()
        self.reward_memory[ndx] = reward
        self.new_th_memory[ndx] = new_state[0]
        self.new_obj_memory[ndx] = new_state[1]
        self.new_jnt_err_memory[ndx] = new_state[2]
        self.new_jnt_dedt_memory[ndx] = new_state[3]
        self.terminal_memory[ndx] = done
        self.time_step[ndx] = time_step
        self.mem_cntr += 1


    def sample_buffer(self, batch_size, batch=None, everything=False, use_batch=False):
        min_mem = min(self.mem_cntr, self.mem_size)
        if everything:
            batch = np.arange(min_mem,dtype=np.int32)
        elif use_batch:
            batch = batch
        else:
            batch = rng.choice(min_mem, batch_size,replace=False)
        
        coord_batch = []
        feat_batch = []
        new_coord_batch = []
        new_feat_batch = []
        coord_list = [] 
        new_coord_list = []
        feat_list = []
        new_feat_list = []
        for b in batch:
            for t in range(self.time_d):
                ndx_i = (b - t) % self.mem_size
                if ndx_i >= 0: # check if ndx is out of range
                    if self.time_step[ndx_i]<=self.time_step[b]: # check if still in same episode
                        th = self.th_memory[ndx_i]
                        new_th = self.new_th_memory[ndx_i]
                        obj_arr = self.obj_memory[ndx_i]
                        new_obj_arr = self.new_obj_memory[ndx_i]
                        rob_coords,rob_feats = get_coords(th,t)
                        new_rob_coords,new_rob_feats = get_coords(new_th,t)
                        coord_list.append(rob_coords)
                        feat_list.append(rob_feats)
                        new_coord_list.append(new_rob_coords)
                        new_feat_list.append(new_rob_feats)
                        r,c = obj_arr.shape
                        for i in range(c):
                            pos = obj_arr[:,i]
                            new_pos = new_obj_arr[:,i]
                            obj_c,obj_f = obj_get_coords(pos,t)
                            new_obj_c,new_obj_f = obj_get_coords(new_pos,t)
                            coord_list.append(obj_c)
                            feat_list.append(obj_f)
                            new_coord_list.append(new_obj_c)
                            new_feat_list.append(new_obj_f)
                    else:
                        t = self.time_d + 1
                else:
                    t = self.time_d + 1
            
            coord_batch.append(np.vstack(coord_list))
            feat_batch.append(np.vstack(feat_list))
            new_coord_batch.append(np.vstack(new_coord_list))
            new_feat_batch.append(np.vstack(new_feat_list))
            feat_list = []
            coord_list = []
            new_coord_list = []
            new_feat_list = []

        jnt_err = self.jnt_err_memory[batch]
        jnt_dedt = self.jnt_dedt_memory[batch]
        new_jnt_err = self.new_jnt_err_memory[batch]
        new_jnt_dedt = self.new_jnt_dedt_memory[batch]
        weights = self.weight_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return (coord_batch, feat_batch, jnt_err, jnt_dedt), weights, actions, \
                rewards, (new_coord_batch, new_feat_batch, new_jnt_err, new_jnt_dedt), dones
        
    def save(self):
        print('saving buffer')
        file_str = self.file + '.pkl'
        with open(file_str, 'wb') as file:
            pickle.dump(self,file)

    def load(self):
        print('loading buffer')
        file_str = self.file + '.pkl'
        with open(file_str, 'rb') as file:
            new_buff = pickle.load(file)

        return new_buff
        # # coord_memory, feat_memory, and jnt_err all define the state 
        # self.coord_memory = new_buff.coord_memory
        # self.feat_memory = new_buff.feat_memory
        # self.jnt_err_memory = new_buff.jnt_err_memory

        # self.new_coord_memory = new_buff.new_coord_memory
        # self.new_feat_memory = new_buff.new_feat_memory
        # self.new_jnt_err_memory = new_buff.new_jnt_err_memory

        # self.action_memory = new_buff.action_memory
        # self.reward_memory = new_buff.reward_memory
        # self.terminal_memory = new_buff.terminal_memory
        # self.time_step = new_buff.time_step 

    def clear(self):
        max_size = self.mem_size
        jnt_d = self.jnt_d
        self.th_memory = np.empty(max_size,dtype=np.object)
        self.obj_memory = np.empty(max_size,dtype=np.object)
        self.jnt_err_memory = np.zeros((self.mem_size,jnt_d))
        self.jnt_dedt_memory = np.zeros((self.mem_size,jnt_d))
        self.new_th_memory = np.empty(max_size,dtype=np.object)
        self.new_obj_memory = np.empty(max_size,dtype=np.object)
        self.new_jnt_err_memory = np.zeros((self.mem_size,jnt_d))
        self.new_jnt_dedt_memory = np.zeros((self.mem_size,jnt_d))


        self.action_memory = np.zeros((self.mem_size, jnt_d))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.time_step = np.ones(self.mem_size) * np.inf
        self.mem_cntr = 0

    def add_data(self, mems):
        for mem in mems:
            for i in range(mem.mem_cntr):
                ndx = self.mem_cntr % self.mem_size
                self.th_memory[ndx] = mem.th_memory[i]
                self.obj_memory[ndx] = mem.obj_memory[i]
                self.jnt_err_memory[ndx] = mem.jnt_err_memory[i]
                self.jnt_dedt_memory[ndx] = mem.jnt_dedt_memory[i]
                self.action_memory[ndx] = mem.action_memory[i]
                self.new_th_memory[ndx] = mem.new_th_memory[i]
                self.new_obj_memory[ndx] = mem.new_obj_memory[i]
                self.new_jnt_err_memory[ndx] = mem.new_jnt_err_memory[i]
                self.new_jnt_dedt_memory[ndx] = mem.new_jnt_dedt_memory[i]
                self.time_step[ndx] = mem.time_step[i]
                self.reward_memory[ndx] = mem.reward_memory[i]
                self.weight_memory[ndx] = mem.weight_memory[i]
                self.terminal_memory[ndx] = mem.terminal_memory[i]

                self.mem_cntr += 1