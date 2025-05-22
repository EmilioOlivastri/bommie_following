#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <ros/ros.h>

#include <iostream>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

void setCameraInfo(const std::string& cam_name, const YAML::Node& config,
                   const std::string& stereo_rig, const std::string& frame_id,
                   const std::string& distortion_model, 
                   const cv::Mat& r_mat, const cv::Mat& p_mat,
                   sensor_msgs::CameraInfo& cam_info);

void getCameraInfo(const std::string& cam_name, const YAML::Node& config, 
                   cv::Mat& k_mat, cv::Mat& d_mat, cv::Mat& trasf_mat);


class StereoUndistorter
{
public:

    StereoUndistorter(const YAML::Node& config);
    
private:

    void undistortLeftCallback(const sensor_msgs::ImageConstPtr& left_img);
    void undistortRightCallback(const sensor_msgs::ImageConstPtr& right_img);

    cv::Mat _map_left_x;
    cv::Mat _map_left_y;
    cv::Mat _map_right_x; 
    cv::Mat _map_right_y;

    sensor_msgs::CameraInfo _cam_info_left;
    sensor_msgs::CameraInfo _cam_info_right;

    ros::NodeHandle _n;
    ros::Subscriber _left_sub;
    ros::Publisher _left_pub_rect;
    ros::Subscriber _right_sub;
    ros::Publisher _right_pub_rect;
    ros::Publisher _info_left_pub;
    ros::Publisher _info_right_pub;
};