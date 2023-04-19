#include <ros/ros.h>
#include <mink_control/StringWithHeader.h>
// #include <eigen.h>

int main(int argc, char** argv) {
    // init ros - declare a name for the node handler
    ros::init(argc, argv, "talker");
    // declare node handle  - primary way of interacting with ROS library 
    ros::NodeHandle node_handle;
    // declare a ros publisher called publisher using node_handle.advertise
    // advertise needs template parameter with message type <> and name of topic we are trying to publish to and qeue size  
    ros::Publisher publisher = node_handle.advertise<mink_control::StringWithHeader>("talker_topic", 1);

    ros::Rate rate(1 /*Hz*/); // a function that allows use to change the frequency of loops
    while(ros::ok) { // will keep going while the ros node is told to shutdown
        // mink_control::StringWithHeader msg; // declare a message
        // publisher.publish(msg); // publish message onto the "talker_topic"
        ros::spinOnce(); //"spinOnce" allows ros to take care of other programs running
        rate.sleep(); // rate.sleep makes sure the while loop completes the loop at 10 Hz freq
    }

    return 0;
}