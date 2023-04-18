#include <ros/ros.h>
#include <mink_control/StringWithHeader.h>
#include <iostream>
#include <Eigen/Geometry>
#include <math.h>
#include <fstream>
#include <string.h>

int freq = 100;
const double dt = 1/(double)freq;
const double dt_2 = pow(dt,2)/2;
const double P = 100;
const double D = 50;
typedef Eigen::Array<double,5,1> JointVector;
JointVector goal;

void callback(const mink_control::StringWithHeaderConstPtr& msg) {
    // when passing in pointers, we don't use . operator. we use -> to point to the original variable and call the method
    goal(0) = msg->jnt1;
    goal(1) = msg->jnt2;
    goal(2) = msg->jnt3;
    goal(3) = msg->jnt4;
    goal(4) = msg->jnt5;
}

const double calc_angle(double s, double c) {
    const double ang = std::atan2(s,c);
    return ang;
}

void write_to_csv(const string file_name, Eigen::Array<double,5,1> arr) {
    std::string s0 = std::to_string(arr[0]); 
    std::string s1 = std::to_string(arr[1]); 
    std::string s2 = std::to_string(arr[2]); 
    std::string s3 = std::to_string(arr[3]); 
    std::string s4 = std::to_string(arr[4]); 
    std::string str_out = s0+","+s1+","+s2+","+s3+","+s4;
    std::cout << str_out << ;

}

int main(int argc, char** argv) {
    std::ofstream myfile;
    myfile.open("/home/jrockholt@ad.ufl.edu/Documents/traj_output.csv");
    myfile<< "I wrote something"<<std::endl;
    myfile.close();
    myfile.open("/home/jrockholt@ad.ufl.edu/Documents/traj_output.csv");
    // myfile.open("traj_output.csv");
    ros::init(argc, argv, "listener");
    ros::NodeHandle node_handle;
    ros::Subscriber subscriber = node_handle.subscribe("talker_topic", 1, &callback);
    ros::Publisher traj_pub = node_handle.advertise<mink_control::StringWithHeader>("way_ptn", 1);
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

    while(ros::ok) {
        // myfile.open("/home/jrockholt@ad.ufl.edu/jay_ws_/src/mink_control/file/traj_output.csv");
        ros::spinOnce();
        
        jnt_err = goal - q;
        JointVector s = jnt_err.sin(); 
        JointVector c = jnt_err.cos();
        for (int i = 0; i<5; i++) { jnt_err[i] = calc_angle(s[i],c[i]); }
        dedt = -1*q_dot;
        q_2dot = P*jnt_err + D*dedt;
        // recalculated based on acceleration limits
        for (int i=0; i<5; i++) {
            if (q_2dot[i] > q_2dot_max[i]) { q_2dot[i] = q_2dot_max[i]; }
            if (q_2dot[i] < -1*q_2dot_max[i]) { q_2dot[i] = -1*q_2dot_max[i]; }
        }
        nxt_q_dot = (q_2dot - Z*q_dot) * dt + q_dot;

        // recalculate q_2dot if jnt_vel is exceeded
        for (int i = 0; i<5; i++) { 
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
        std::string s0 = std::to_string(nxt_q[0]); 
        std::string s1 = std::to_string(nxt_q[1]); 
        std::string s2 = std::to_string(nxt_q[2]); 
        std::string s3 = std::to_string(nxt_q[3]); 
        std::string s4 = std::to_string(nxt_q[4]); 
        myfile << s0+","+s1+","+s2+","+s3+","+s4+"\n" << std::endl;
        mink_control::StringWithHeader way_ptn;
        way_ptn.jnt1 = nxt_q[0]; way_ptn.jnt2 = nxt_q[1]; way_ptn.jnt3 = nxt_q[2];
        way_ptn.jnt4 = nxt_q[3]; way_ptn.jnt5 = nxt_q[4];
        q = nxt_q; q_dot = nxt_q_dot;
        std::string str_out = s0+","+s1+","+s2+","+s3+","+s4;
        std::cout << str_out << std::endl;
        traj_pub.publish(way_ptn);

        rate.sleep();
        // if (jnt_err.isApproxToConstant(0.0) and dedt.isApproxToConstant(0.0)) {
        //     std::cout<<"reached goal"<<std::endl;
        //     ros::spin();
        // }
    }

    myfile.close();
    return 0;
}