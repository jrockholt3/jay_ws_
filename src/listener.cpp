#include <ros/ros.h>
#include <mink_control/StringWithHeader.h>
#include <iostream>
#include <Eigen/Geometry>
#include <math.h>
#include <fstream>
#include <string.h>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/JointState.h>

int freq = 1000;
const int num_jnt = 6;
const double dt = 1/(double)freq;
const double dt_2 = dt*dt; //pow(dt,2)/2;
const double P = 10;
const double D = 5;
typedef Eigen::Array<double,5,1> JointVector;
JointVector goal;
JointVector curr;

void callback(const mink_control::StringWithHeaderConstPtr& msg) {
    // when passing in pointers, we don't use . operator. we use -> to point to the original variable and call the method
    goal(0) = msg->jnt1;
    goal(1) = msg->jnt2;
    goal(2) = msg->jnt3;
    goal(3) = msg->jnt4;
    goal(4) = msg->jnt5;
}

void jnt_state_callback(const sensor_msgs::JointStateConstPtr& msg) {
    curr[0] = msg->position[0];
    curr[1] = msg->position[1];
    curr[2] = msg->position[2];
    curr[3] = msg->position[3];
    curr[4] = msg->position[4];
}

Eigen::Array<double, 5, 1> calc_angle(Eigen::Array<double, 5, 1> th) {
    JointVector s = th.sin();
    JointVector c = th.cos();

    for (int i=0; i<5; i++) {
        double s_i = s[i]; double c_i = c[i];
        th[i] = std::atan2(s_i,c_i);
    }
    return th;
}


class csv_writer {
// this class just makes a simple csv_writer to write joint vectors to csv files 
    private:
        std::ofstream this_file;
        std::string file_path;
    public:
        csv_writer(std::string in_file_path) {
            this->file_path = in_file_path;
            this->open_file();
        }

        void open_file() {
            this_file.open(this->file_path);
        }

        void write_JointVect(Eigen::Array<double,5,1> nxt_q) {
            std::string s0 = std::to_string(nxt_q[0]); 
            std::string s1 = std::to_string(nxt_q[1]); 
            std::string s2 = std::to_string(nxt_q[2]); 
            std::string s3 = std::to_string(nxt_q[3]); 
            std::string s4 = std::to_string(nxt_q[4]); 
            this_file << s0+","+s1+","+s2+","+s3+","+s4+"\n" << std::endl;
        }

        void close_file() {
            this_file.close();
        }
};


int main(int argc, char** argv) {
    // std::string file_path0 = "/home/jrockholt@ad.ufl.edu/Documents/git/FiveLinkRobot/traj_out.csv";
    // std::string file_path1 = "/home/jrockholt@ad.ufl.edu/Documents/git/FiveLinkRobot/goal_recieved.csv";
    std::string file_path0 = "/home/jrockholt@ad.ufl.edu/Documents/traj_out.csv";
    std::string file_path1 = "/home/jrockholt@ad.ufl.edu/Documents/goal_recieved.csv";
    csv_writer csv0(file_path0);
    csv_writer csv1(file_path1);

    // myfile.open("/home/jrockholt@ad.ufl.edu/Documents/traj_output.csv");
    // myfile<< "I wrote something"<<std::endl;
    // myfile.close();
    // myfile.open("/home/jrockholt@ad.ufl.edu/Documents/traj_output.csv");
    // myfile.open("traj_output.csv");
    ros::init(argc, argv, "listener");
    ros::NodeHandle node_handle;
    ros::Subscriber jnt_state_sub = node_handle.subscribe("joint_states", 1, jnt_state_callback);
    ros::Subscriber subscriber = node_handle.subscribe("talker_topic", 1, &callback);
    // ros::Publisher updated_pub = node_handle.advertice<std_msgs::Bool>("update_status",1)
    ros::Publisher traj_pub = node_handle.advertise<sensor_msgs::JointState>("joint_states",10);
    ros::Rate rate(freq);

    JointVector jnt_err;
    JointVector dedt;
    JointVector q;
    JointVector q_dot;
    JointVector q_2dot;
    JointVector Z;
    JointVector q_2dot_max;
    JointVector q_dot_max;
    JointVector nxt_q;
    JointVector nxt_q_dot;

    goal.setZero();
    q.setZero();
    q_dot.setZero();
    const double a = 1.5; const double b = 0.5; const double c = 1.5;
    q_2dot_max.setConstant(a);
    Z = b*q_2dot_max;
    q_dot_max.setConstant(c);
    sensor_msgs::JointState q_out;
    q_out.name.resize(num_jnt);
    q_out.position.resize(num_jnt);
    q_out.velocity.resize(num_jnt);
    q_out.effort.resize(num_jnt);
    q_out.name[0] = 

    bool boo = true;
    while ((ros::ok()) && (boo)) {
        // myfile.open("/home/jrockholt@ad.ufl.edu/jay_ws_/src/mink_control/file/traj_output.csv");
        q = curr;
        goal = calc_angle(goal);
        q = calc_angle(q);
        jnt_err = goal - q;
        dedt = -1*q_dot;
        q_2dot = P*jnt_err + D*dedt;
        // recalculated based on acceleration limits
        for (int i=0; i<5; i++) {
            if (ros::ok() == false) { boo = false; };
            if (q_2dot[i] > q_2dot_max[i]) { q_2dot[i] = q_2dot_max[i]; }
            if (q_2dot[i] < -1*q_2dot_max[i]) { q_2dot[i] = -1*q_2dot_max[i]; }
        }

        nxt_q_dot = (q_2dot - Z*q_dot) * dt + q_dot;

        // recalculate q_2dot if jnt_vel is exceeded
        for (int i=0; i<5; i++) { 
            if (ros::ok() == false) { boo = false; };
            if (nxt_q_dot[i] > q_dot_max[i]) { 
                nxt_q_dot[i] = q_dot_max[i];
                q_2dot[i] = (nxt_q_dot[i] - q_dot[i])/dt + Z[i]*q_dot[i];
            }
            if (nxt_q_dot[i] < -1*q_dot_max[i]) { 
                nxt_q_dot[i] = -1*q_dot_max[i];
                q_2dot[i] = (nxt_q_dot[i] - q_dot[i])/dt + Z[i]*q_dot[i];
            } 
        }

        nxt_q = (q_2dot - Z*q_dot)*dt_2 + q_dot*dt + q;
        for (int i=0; i<5; i++) {
            q_out.position[i] = nxt_q[i];
        }
        traj_pub.publish(q_out);

        if (ros::ok()) { csv0.write_JointVect(nxt_q); }
        if (ros::ok()) { csv1.write_JointVect(goal); }


        q = nxt_q; q_dot = nxt_q_dot;
        // std::string str_out = s0+","+s1+","+s2+","+s3+","+s4;
        // std::cout << goal << std::endl;
        // traj_pub.publish(way_ptn);

        ros::spinOnce();
        rate.sleep();
    }

    csv0.close_file();
    csv1.close_file();
    return 0;
}