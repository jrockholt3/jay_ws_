import torch 
import MinkowskiEngine as ME
import numpy as np
    
def act_preprocessing(state, weights, single_value=False,device='cuda'):
    if single_value:
        coords,feats = ME.utils.sparse_collate([state[0]],[state[1]])
        jnt_err = state[2]#.clone().detach()
        jnt_err = torch.tensor(jnt_err,dtype=torch.double,device=device).view(1,state[2].shape[0])
        jnt_dedt = state[3]
        jnt_dedt = torch.tensor(jnt_dedt,dtype=torch.double,device=device).view(1,state[3].shape[0])
        w = torch.tensor(weights,dtype=torch.double,device=device).view(1,weights.shape[0])
    else:
        coords,feats = ME.utils.sparse_collate(state[0],state[1])
        jnt_err = state[2]#.clone().detach()
        jnt_err = torch.tensor(jnt_err,dtype=torch.double,device=device)
        jnt_dedt = state[3]
        jnt_dedt = torch.tensor(jnt_dedt,dtype=torch.double,device=device)
        w = torch.tensor(weights,dtype=torch.double,device=device)

    x = ME.SparseTensor(coordinates=coords, features=feats.double(),device=device)
    return x, jnt_err, jnt_dedt, w

def crit_preprocessing(state, weights, action, single_value=False, device='cuda'):
    if single_value:
        coords,feats = ME.utils.sparse_collate([state[0]],[state[1]])
        jnt_err = state[2]#.clone().detach()
        jnt_err = torch.tensor(jnt_err,dtype=torch.double,device=device).view(1,state[2].shape[0])
        jnt_dedt = state[3]
        jnt_dedt = torch.tensor(jnt_dedt,dtype=torch.double,device=device).view(1,state[3].shape[0])
        w = torch.tensor(weights,dtype=torch.double,device=device).view(1,weights.shape[0])
        a = torch.tensor(action,dtype=torch.double,device=device).view(1,action.shape[0])
    else:
        coords,feats = ME.utils.sparse_collate(state[0],state[1])
        jnt_err = state[2]#.clone().detach()
        jnt_err = torch.tensor(jnt_err,dtype=torch.double,device=device)
        jnt_dedt = state[3]
        jnt_dedt = torch.tensor(jnt_dedt,dtype=torch.double,device=device)
        w = torch.tensor(weights,dtype=torch.double,device=device)
        a = torch.tensor(action,dtype=torch.double,device=device)

    x = ME.SparseTensor(coordinates=coords, features=feats.double(),device=device)
    return x, jnt_err, jnt_dedt, w, a

def create_stack(state:tuple):
    coord_list = []
    feat_list = []
    c_arr = state[0]
    for j in range(6):
        c_arr[:,0] = j
        coord_list.append(c_arr.copy())
        feat_list.append(state[1])
    return coord_list, feat_list

def stack_arrays(coord_list:list, feat_list:list, new_state:tuple):
    '''
    inputs:
        coord_list : the running list of vertically stacked np.arrays of point cloud coordinates
        feat_list : the running list of vertically stacked np.arrays of corresponding feature values
        new_state : the newest environement state (coord_list, feat_list, jnt_err, jnt_dedt)
    outputs:
        coord_list : a list of vertically stacked np.arrays with the newest coord array inserted at 
                     index 0 and the lastest removed
        feat_list : feat array for the correpsonding coordinate
    '''
    coord_list.insert(0,new_state[0])
    feat_list.insert(0,new_state[1])
    if len(coord_list) > 6:
        coord_list.pop()
        feat_list.pop()
    new_list = []
    for j,c in enumerate(coord_list):
        c[:,0] = j
        new_list.append(c.copy())
    coord_list = new_list.copy()

    return coord_list, feat_list


# def gen_obstacles(env:RobotEnv, obs_index:index.Index):
#     '''
#     have to generate the min and max corners of obstacles at 
#     each time step
#     '''
#     objs = env.objs.copy()
#     dt = Robot_Env.dt
#     t_limit = Robot_Env.t_limit
#     min_prox = Robot_Env.min_prox
#     obs = []

#     t = 0
#     time_steps = int(np.ceil(t_limit/dt))
#     while t <= time_steps:
#         for o in objs:
#             center = o.curr_pos
#             x,y,z = center[0],center[1],center[2]
#             obs_i = (t, x-min_prox, y-min_prox, z-min_prox, t, x+min_prox, y+min_prox, z+min_prox)
#             obs_index.add(uuid.uuid4(), obs_i)
#             obs.append(obs_i)
#             o.step()
#         t += 1

#     return obs

        