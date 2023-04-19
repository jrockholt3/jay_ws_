import rospy
import std_msgs
# from std_msgs.msg import String
from mink_control.msg import StringWithHeader
import numpy as np
import pickle
import csv
from sensor_msgs.msg import JointState

# def callback(data):
#     rospy.loginfo(rospy.get_caller_id + "I heard", data.data)
jnt_vel_max = 1.5 #np.ones(5)*1.5
pub_freq = 2000
update_freq = 50
dt = 1/update_freq
q_2dot_max = np.pi # rad/s^2
Z = .5*q_2dot_max
q = np.zeros(5, dtype=np.float64)
q_dot = np.zeros(5,dtype=np.float64)

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

def jnt_state_callback(state):
    q = state.position
    q_dot = state.velocity
    # print("py_talker: updated state")
    


def talker():
    pub = rospy.Publisher('talker_topic',StringWithHeader,queue_size=10)
    rospy.Subscriber('joint_states', JointState, jnt_state_callback)
    # sub = rospy.Subscriber('updated_status',std_msgs.bool,queue_size=1)
    rospy.init_node('py_talker',anonymous=True)
    rate = rospy.Rate(pub_freq)
    # file_path = "/home/jrockholt@ad.ufl.edu/Documents/git/FiveLinkRobot/goal_published.csv"
    file_path = "/home/jrockholt@ad.ufl.edu/Documents/goal_published.csv";
    file = open(file_path)
    csv_writer = csv.writer(file)

    # file = open('/home/jrockholt@ad.ufl.edu/Documents/git/FiveLinkRobot/action_mem.pkl','rb')
    # actions = pickle.load(file)
    actions = np.zeros((150,5),dtype=float)
    max_i = actions.shape[0]

    i = 0
    j = 40
    msg = StringWithHeader()
    msg.jnt1 = q[0].item()
    msg.jnt2 = q[1].item()
    msg.jnt3 = q[2].item()
    msg.jnt4 = q[3].item()
    msg.jnt5 = q[4].item()
    
    rate1 = rospy.Rate(0.1)
    rate1.sleep()
    while not rospy.is_shutdown():
        while i < max_i and not rospy.is_shutdown():

            if j%40 == 0:
                if i%100==0: print('still running', i)
                action_i = actions[i,:]
                # print('action', action_i)
                # nxt_q, nxt_q_dot = nxt_state(q, q_dot, action_i)

                msg = StringWithHeader()
                # msg.jnt1 = nxt_q[0].item()
                # msg.jnt2 = nxt_q[1].item()
                # msg.jnt3 = nxt_q[2].item()
                # msg.jnt4 = nxt_q[3].item()
                # msg.jnt5 = nxt_q[4].item()
                msg.jnt1 = np.pi/4
                msg.jnt2 = np.pi/4
                msg.jnt3 = np.pi/4
                msg.jnt4 = np.pi/4
                msg.jnt5 = np.pi/4

                # q = nxt_q
                # q_dot = nxt_q_dot
                i += 1
                j = 0
            j+=1
            
            pub.publish(msg)
            # str_out = str(msg.jnt1)+","+str(msg.jnt2)+","+str(msg.jnt3)+","+str(msg.jnt4)+","+str(msg.jnt5)
            # csv_writer.writerow(str_out)
            rate.sleep()
        print("done reading", i)
        rospy.spin()



if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass