#!/usr/bin/env python

import rospy
import std_msgs
# from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray, Float32MultiArray
import numpy as np
import pickle
import csv
from sensor_msgs.msg import JointState
import warnings
warnings.filterwarnings('ignore')
import torch 
from mink_control.import_me_if_you_can import say_it_works
from mink_control.Object_v2 import obj_get_coords
from mink_control.Robot_5link_Env import RobotEnv, gen_rand_pos
from mink_control.Robot_5link import get_coords, S, a, l, reverse
from mink_control.sparse_tnsr_replay_buffer import ReplayBuffer
import numpy as np
from time import time, sleep
from mink_control.utils import act_preprocessing, stack_arrays
from mink_control.env_config import *
from mink_control.optimized_functions import calc_jnt_err, PDControl
from mink_control.optimized_functions_5L import nxt_state
from mink_control.Networks import SupervisedActor
import subprocess

jnt_vel_max = 1.5 #np.ones(5)*1.5
pub_freq = 2000
update_freq = int(np.round(1/dt))
loop_count = int(np.floor(pub_freq/update_freq))
print('loop_count',loop_count)
# dt = 1/update_freq
q_2dot_max = tau_max
# Z = .5*q_2dot_max
q = np.zeros(5, dtype=np.float64)
q_dot = np.zeros(5,dtype=np.float64)
obj1 = np.ones(3)*0.5
obj2 = np.ones(3)*100
obj3 = np.ones(3)*100
last_t = time()
dt = 1

def angle_calc(th_arr):
    s = np.sin(th_arr)
    c = np.cos(th_arr)
    th = np.arctan2(s,c)
    return th

# def nxt_state(q, q_dot, action):
#     nxt_q_dot = (action-Z*q_dot)*dt + q_dot
#     flag = False
#     if np.any(nxt_q_dot > jnt_vel_max):
#         flag = True 
#         nxt_q_dot[nxt_q_dot > jnt_vel_max] = jnt_vel_max #jnt_vel_max[nxt_q_dot > jnt_vel_max]
#     if np.any(nxt_q_dot < -1*jnt_vel_max):
#         nxt_q_dot[nxt_q_dot > jnt_vel_max] = -1*jnt_vel_max #[nxt_q_dot > jnt_vel_max]
#         flag = True 
#     if flag:
#         action = (nxt_q_dot - q_dot) / dt + Z*q_dot
#     nxt_q = (action-Z*q_dot)*dt**2/2 + q_dot*dt + q
#     nxt_q = angle_calc(nxt_q)
#     return (nxt_q, nxt_q_dot)

# need to make something that reads in the objects location then makes a point cloud that can be used for the model.
#  The model then outputs an action and the goal is updated
def read_scene(msg):
    global obj1, obj2, obj3 
    data = msg.data
    data = np.array(data).reshape((3,18))
    # data = [0:pelvis, 1:spine top, 2:necktop, 3:L shoulder, 
    #         4:L elbow, 5:L wrist, 6:R should, 7:R elbow, 8:R wirst ... useless]
    obj1 = data[:,2]
    obj2 = data[:,5]
    obj3 = data[:,8]

# def jnt_state_callback(msg):
#     global q, q_dot, last_t
#     t = time()
#     # print('q', msg.position)
#     # print('w', msg.velocity)
#     th = msg.position
#     w = msg.velocity
#     # th = np.array(th)
#     # th = th[0:5]
#     # w = (th - q)/(t-last_t)
#     if len(w) > 0:
#         for i in range(5):
#             q[i] = th[i]
#             q_dot[i] = w[i]
#     last_t = t

def gen_point_cloud():
    # creating sparse tensor
    c_list,f_list = [],[]
    rob_coords, rob_feats = get_coords(q,time_step=0)
    c_list.append(rob_coords)
    f_list.append(rob_feats)
    o_list = [obj1, obj2, obj3]
    for o in o_list:
        o_c, o_f = obj_get_coords(o,t=0)
        if not o_f.size==0:
            c_list.append(o_c)
            f_list.append(o_f)

    return np.vstack(c_list), np.vstack(f_list)


def talker():
    global q, q_dot, obj1, obj2, obj3
    pub = rospy.Publisher('talker_topic',Float64MultiArray,queue_size=10)
    # human location
    human_loc_sub = rospy.Subscriber('/skeleton_points',Float32MultiArray,read_scene)
    # current location of the robot 
    # jnt_state_sub = rospy.Subscriber('joint_states', JointState, jnt_state_callback)
    # sub = rospy.Subscriber('updated_status',std_msgs.bool,queue_size=1)
    rospy.init_node('py_talker',anonymous=True)
    rate = rospy.Rate(pub_freq)
    net_dir = '/home/jrockholt@ad.ufl.edu/base_workspace/src/mink_control/files/tmp'
    actor = SupervisedActor(name='chckptn_supervised_actor_0509',device='cpu', chckpt_dir=net_dir)
    actor.load_checkpoint()
    # env = RobotEnv()
    # env.start = np.array([0.0,0.0,0.0,0.0,0.0])
    # env.th = env.start

    # defining the start and end joint positions of the robot 
    s,u = gen_rand_pos(1, S, l)
    start = reverse(s,u,S,a,l)
    start = start[0,:]
    sol_found = False
    while not sol_found:
        if np.any(np.isnan(start)):
            s,u = gen_rand_pos(1,S,l)
            start = reverse(s,u,S,a,l)
            start = start[0,:]
        else:
            sol_found = True 
    q = start
    g,v = gen_rand_pos(3,S,l)
    goal = reverse(g,v,S,a,l)
    goal = goal[0,:]
    sol_found = False
    while not sol_found:
        if np.any(np.isnan(goal)):
            g,v = gen_rand_pos(3,S,l)
            goal = reverse(g,v,S,a,l)
            goal = goal[0,:]
        else:
            sol_found = True 

    msg = Float64MultiArray()
    for i in range(5):
        msg.data.append(start[i])
    msg.data.append(0.0)
    pub.publish(msg)

    # defining starting state
    done = False
    t = 0
    jnt_err = calc_jnt_err(start, goal)
    jnt_dedt = np.zeros(5)
    weights = np.random.rand(3)
    c_arr, f_arr = gen_point_cloud()
    coord_list, feat_list = [c_arr],[f_arr]
    state = (coord_list, feat_list, jnt_err, jnt_dedt)
    r = np.linalg.norm(np.ones(5)*jnt_vel_max/5)
    use_PID=False
    pid_steps = 0
    actor_steps = 0
    loop = 0
    print('goal is', goal)
    print('starting',start)
    print('sleeping')
    sleep(5)
    # subprocess.call(args='source devel/setup.bash; roslaunch human_collision_objects fake_human.launch')
    print('starting')
    use_PID = True
    while not done and not rospy.is_shutdown():
        if loop == loop_count:
            # ts = time()
            loop = 0
            jnt_err = calc_jnt_err(q, goal)
            # print('jnt_err time', time()-ts)
            # print('time', time()-ts, 'jnt_err',np.round(jnt_err*180/np.pi,2))
            jnt_dedt = -1*q_dot
            c_temp, f_temp = gen_point_cloud()
            # print('point cloud', time()-ts)
            state = (c_temp, f_temp, jnt_err, jnt_dedt)
            coord_list, feat_list = stack_arrays(coord_list, feat_list, state)
            state_ = (np.vstack(coord_list), np.vstack(feat_list), state[2], state[3])
            x, jnt_err_tnsr, jnt_dedt, w = act_preprocessing(state_, weights,single_value=True,device=actor.device)
            # print('preprocess time', time()-ts)
            if not use_PID:
                with torch.no_grad():
                    ts = time()
                    action = actor.forward(x, jnt_err_tnsr, jnt_dedt,w)
                    # print('forward pass', time()-ts)
                    action = action.numpy()
                    action = action.astype(np.float64)
                    action = action.reshape(5)
            else:
                action = PDControl(state[2],state[3])
            package = nxt_state(np.array([obj1,obj2,obj3]), q, q_dot, action, a, l, S)
            nxt_q = package[0:5,0]
            nxt_q_dot = package[0:5,1]
            prox = package[-1,0]
            # print('nxt_state', time()-ts)
            
            if np.linalg.norm(jnt_err) < r:
                pid_steps += 1
                use_PID = True
            else:
                actor_steps += 1
            
            msg.data.clear()
            for i in range(5):
                msg.data.append(nxt_q[i])
            msg.data.append(0.0)
            pub.publish(msg)
            q = nxt_q
            q_dot = nxt_q_dot
            print('prox',np.round(prox,2),np.round(obj1,2))
        else:
            pub.publish(msg)
            loop += 1
        if np.all(np.round(jnt_err,2) == 0.0):
            done = True 
        else:
            rate.sleep()
    print('spinning')
    rospy.spin()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass