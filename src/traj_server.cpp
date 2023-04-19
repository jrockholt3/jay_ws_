#include <ros/ros.h>
#include <mink_control/StringWithHeader.h>
#include <iostream>
#include <Eigen/Geometry>
#include <math.h>
#include <fstream>
#include <string.h>
#include <std_msgs/Float32MultiArray.h>
#include <mink_control/update_goal.h>


typedef Eigen::Array<float,5,1> JointVector;
JointVector goal;
const int freq = 100;
const float dt = 1/(float)freq;
const float dt_2 = dt*dt; //pow(dt,2)/2;
const float P = 10;
const float D = 5;

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

        void write_JointVect(Eigen::Array<float,5,1> nxt_q) {
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

Eigen::Array<float, 5, 1> calc_angle(Eigen::Array<float, 5, 1> th) {
    JointVector s = th.sin();
    JointVector c = th.cos();

    for (int i=0; i<5; i++) {
        float s_i = s[i]; float c_i = c[i];
        th[i] = std::atan2(s_i,c_i);
    }
    return th;
}


bool goal_update(mink_control::update_goal::Request &req, mink_control::update_goal::Response &res) {
    std::vector<float> arr;
    goal[0] = req.jnt1;
    goal[1] = req.jnt2;
    goal[2] = req.jnt3;
    goal[3] = req.jnt4;
    goal[4] = req.jnt5;
    res.recieved = true;
    std::cout << "goal_update ran" << std::endl;
    return true;
};

int main(int argc, char **argv) {
    std::cout <<"running"<<std::endl;
    ros::init(argc, argv, "traj_server_node");
    ros::NodeHandle n;
    ros::ServiceServer service = n.advertiseService("traj_server", goal_update);
    ros::Publisher traj_pub = n.advertise<mink_control::StringWithHeader>("way_ptn", 10);
    ros::Rate rate(freq);
    ros::spinOnce();

    std::string file_path0 = "/home/jrockholt@ad.ufl.edu/Documents/git/FiveLinkRobot/traj_out.csv";
    std::string file_path1 = "/home/jrockholt@ad.ufl.edu/Documents/git/FiveLinkRobot/goal_recieved.csv";
    csv_writer csv0(file_path0);
    csv_writer csv1(file_path1);
    
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
    q_2dot_max.setConstant(1.5);
    Z = q_2dot_max/2;
    q_dot_max.setConstant(1.5);
    bool boo = false;
    while (ros::ok && boo) {
        goal = calc_angle(goal);
        q = calc_angle(q);
        jnt_err = goal-q;
        dedt = -1*q_dot;
        q_2dot = P*jnt_err + D*dedt;
        for (int i=0; i<5; i++) {
            if (ros::ok() == false) { boo = false; };
            if (q_2dot[i] > q_2dot_max[i]) { q_2dot[i] = q_2dot_max[i]; }
            if (q_2dot[i] < -1*q_2dot_max[i]) { q_2dot[i] = -1*q_2dot_max[i]; }
        }

        nxt_q_dot = (q_2dot - Z*q_dot) * dt + q_dot;

        // recalculate q_2dot if jnt_vel is exceeded
        for (int i = 0; i<5; i++) { 
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
        if (ros::ok()) { csv0.write_JointVect(nxt_q); }
        if (ros::ok()) { csv1.write_JointVect(goal); }
        
        std::string s0 = std::to_string(nxt_q[0]); 
        std::string s1 = std::to_string(nxt_q[1]); 
        std::string s2 = std::to_string(nxt_q[2]); 
        std::string s3 = std::to_string(nxt_q[3]); 
        std::string s4 = std::to_string(nxt_q[4]); 
        // myfile << s0+","+s1+","+s2+","+s3+","+s4+"\n" << std::endl;
        mink_control::StringWithHeader way_ptn;
        way_ptn.jnt1 = nxt_q[0]; way_ptn.jnt2 = nxt_q[1]; way_ptn.jnt3 = nxt_q[2];
        way_ptn.jnt4 = nxt_q[3]; way_ptn.jnt5 = nxt_q[4];
        q = nxt_q; q_dot = nxt_q_dot;
        std::string str_out = s0+","+s1+","+s2+","+s3+","+s4;
        std::cout << goal << std::endl;
        traj_pub.publish(way_ptn);

        ros::spinOnce();
        rate.sleep();
    }

    csv0.close_file();
    csv1.close_file();
    return 0;
}