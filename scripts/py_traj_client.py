import rospy
import std_msgs
# from std_msgs.msg import String
from mink_control.msg import StringWithHeader
from mink_control.srv import update_goal
import numpy as np
import pickle

from mink_control.srv import update_goal

def angle_calc(th_arr):
    s = np.sin(th_arr)
    c = np.cos(th_arr)
    th = np.arctan2(s,c)
    return th

def nxt_state(q, q_dot, action):
    nxt_q_dot = (action-Z*q_dot)*dt + q_dot
    flag = False
    if np.any(nxt_q_dot > jnt_vel_max):
        flag = True 
        nxt_q_dot[nxt_q_dot > jnt_vel_max] = jnt_vel_max #jnt_vel_max[nxt_q_dot > jnt_vel_max]
    if np.any(nxt_q_dot < -1*jnt_vel_max):
        nxt_q_dot[nxt_q_dot > jnt_vel_max] = -1*jnt_vel_max #[nxt_q_dot > jnt_vel_max]
        flag = True 
    if flag:
        action = (nxt_q_dot - q_dot) / dt + Z*q_dot
    nxt_q = (action-Z*q_dot)*dt**2/2 + q_dot*dt + q
    nxt_q = angle_calc(nxt_q)
    return (nxt_q, nxt_q_dot)

def request(req):
    return 1

# def run_service(goal):
#     rospy.wait_for_service('traj_server')
#     try:
#         msg = StringWithHeader()
#         msg.jnt1 = goal[0]
#         msg.jnt2 = goal[1]
#         msg.jnt3 = goal[2]
#         msg.jnt4 = goal[3]
#         msg.jnt5 = goal[4]
#         return traj_server(msg)
#     except rospy.ServiceException as e:
        # print("Service call failed: %s"%e)

def talker():
    file = open('/home/jrockholt@ad.ufl.edu/Documents/git/FiveLinkRobot/action_mem.pkl','rb')
    # pub = rospy.Publisher('talker_topic',StringWithHeader,queue_size=10)
    # sub = rospy.Subscriber('updated_status',std_msgs.bool,queue_size=1)
    rospy.init_node('py_traj_client',anonymous=True)
    traj_server = rospy.ServiceProxy('traj_server',update_goal)
    actions = pickle.load(file)
    rate = rospy.Rate(60)
    max_i = actions.shape[0]
    q = np.zeros(5, dtype=np.float64)
    q_dot = np.zeros(5,dtype=np.float64)
    nxt_q = np.ones(5)*np.pi/4

    i = 0
    while not rospy.is_shutdown():
        while i < max_i and not rospy.is_shutdown():
            if i%100==0: print('still running', i)
            action_i = actions[i,:]
            # print('action', action_i)
            # nxt_q, nxt_q_dot = nxt_state(q, q_dot, action_i)

            rospy.wait_for_service('traj_server')
            # print('got past rospy.wait')
            traj_server(nxt_q[0],nxt_q[1],nxt_q[2],nxt_q[3],nxt_q[4])
            # msg = StringWithHeader()
            # # msg.jnt1 = nxt_q[0].item()
            # # msg.jnt2 = nxt_q[1].item()
            # # msg.jnt3 = nxt_q[2].item()
            # # msg.jnt4 = nxt_q[3].item()
            # # msg.jnt5 = nxt_q[4].item()
            # msg.jnt1 = np.pi/4
            # msg.jnt2 = np.pi/4
            # msg.jnt3 = np.pi/4
            # msg.jnt4 = np.pi/4
            # msg.jnt5 = np.pi/4
            # pub.publish(msg)


            # q = nxt_q
            # q_dot = nxt_q_dot
            i += 1
            rate.sleep()
        print("done reading", i)
        rate.sleep()



if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass