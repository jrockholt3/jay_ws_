import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from numpy import pi 
from math import pi,tau,dist,fabs,cos,sqrt

def dist(p,q):
    return sqrt(sum((p_i-q_i)**2.0 for p_i,q_i in zip(p,q)))

def all_close(goal, actual, tolerance):

    if type(goal) is list:
        for i in range(len(goal)):
            if abs(actual[i] - goal[i]) > tolerance:
                return False
    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose,actual.pose,tolerance)
    elif type(goal) is geometry_msgs.msg.Pose:
        x0,y0,z0,qx0,qy0,qz0,qw0 = pose_to_list(actual)
        x1,y1,z1,qx1,qy1,qz1,qw1 = pose_to_list(goal)
        d = dist((x1,y1,z1),(x0,y0,z0))
        cos_phi_half = fabs(qx0*qx1 + qy0*qy1 + qz0*qz1 + qw0*qw1)
        return d<= tolerance and cos_phi_half >= cos(tolerance/2.0)
    
    return True 

class MoveGroupTut(object):

    def __init__(self):
        # setup
        super(MoveGroupTut, self).__init__()

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_test",anonymous=True)

        robot = moveit_commander.RobotCommander()

        scene = moveit_commander.PlanningSceneInterface()

        group_name = 'edo'
        move_group = moveit_commander.MoveGroupCommander(group_name)

        # basic info
        planning_frame = move_group.get_planning_frame()
        print("Planning frame: %s" % planning_frame)
        eef_link = move_group.get_end_effector_link()
        group_names = robot.get_group_names()
        print("Planning Groups", robot.get_group_names())

        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        # self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

    def go_to_joint_state(self):
        joint_goal = self.move_group.get_current_joint_values()
        th = pi/4
        for i in range(len(joint_goal)):
            joint_goal[i] = th
        self.move_group.go(joint_goal,wait=True)
        self.move_group.stop()
        current_joints = self.move_group.get_current_joint_values()
        print(self.move_group.get_current_pose().pose)
        return all_close(joint_goal,current_joints, 0.01)
    
    def go_to_pose_goal(self):
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.w = 1.0
        pose_goal.orientation.x = 0.0
        pose_goal.orientation.y = 0.0
        pose_goal.orientation.z = 0.0
        pose_goal.position.x = 0.0
        pose_goal.position.y = 0.0
        pose_goal.position.z = 1.0
        self.move_group.set_pose_target(pose_goal)
        succ = self.move_group.go(wait=True)
        print('succ',succ)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        curr_pos = self.move_group.get_current_pose().pose
        return all_close(pose_goal, curr_pos, 0.01)

def main():
    print('main ran')
    tut = MoveGroupTut()
    tut.go_to_joint_state()
    print(tut.go_to_pose_goal())

if __name__ == "__main__":
    print('i ran')
    main()