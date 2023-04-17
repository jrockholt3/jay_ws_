import rospy
# from std_msgs.msg import String
from mink_control.msg import StringWithHeader
import numpy as np
import pickle


# def callback(data):
#     rospy.loginfo(rospy.get_caller_id + "I heard", data.data)
jnt_vel_max = np.ones(5)*np.pi/2
freq = 2
dt = 1/freq
q_2dot_max = np.pi # rad/s^2
Z = .5*q_2dot_max

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
        nxt_q_dot[nxt_q_dot > jnt_vel_max] = jnt_vel_max[nxt_q_dot > jnt_vel_max]
    if np.any(nxt_q_dot < -1*jnt_vel_max):
        nxt_q_dot[nxt_q_dot > jnt_vel_max] = -1*jnt_vel_max[nxt_q_dot > jnt_vel_max]
        flag = True 
    if flag:
        action = (nxt_q_dot - q_dot) / dt + Z*q_dot
    nxt_q = (action-Z*q_dot)*dt**2/2 + q_dot*dt + q
    nxt_q = angle_calc(nxt_q)
    return (nxt_q, nxt_q_dot)


def talker():
    file = open('/home/jrockholt@ad.ufl.edu/jay_ws_/src/mink_control/files/action_mem.pkl','rb')
    pub = rospy.Publisher('talker_topic',StringWithHeader,queue_size=1)
    rospy.init_node('py_talker',anonymous=True)
    actions = pickle.load(file)
    rate = rospy.Rate(60)
    max_i = actions.shape[0]
    q = np.zeros(5, dtype=np.float64)
    q_dot = np.zeros(5,dtype=np.float64)


    while not rospy.is_shutdown():
        i = 0
        while i < max_i:
            action_i = actions[i,:]
            nxt_q, nxt_q_dot = nxt_state(q, q_dot, action_i)

            msg = StringWithHeader()
            msg.jnt1 = nxt_q[0].item()
            msg.jnt2 = nxt_q[1].item()
            msg.jnt3 = nxt_q[2].item()
            msg.jnt4 = nxt_q[3].item()
            msg.jnt5 = nxt_q[4].item()
            pub.publish(msg)

            q = nxt_q
            q_dot = nxt_q_dot
            i += 1
            rate.sleep()
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass