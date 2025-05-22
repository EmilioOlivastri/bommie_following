#include "bommie_following/stereo_undistorter.hpp"

StereoUndistorter::StereoUndistorter(const YAML::Node& config)
{
    
    std::string node_name = ros::this_node::getName();
    std::string namespace_name = ros::this_node::getNamespace();
    std::string camera_name, stereo_rig, frame_id, distortion_model;
    _n.getParam(node_name + "/stereo_rig", stereo_rig);
    _n.getParam(node_name + "/camera_name", camera_name);
    _n.getParam(node_name + "/frame_id", frame_id);
    _n.getParam(node_name + "/distortion_model", distortion_model);

    _left_sub = _n.subscribe(namespace_name + "/" + stereo_rig  + "/left/image_raw", 1000, &StereoUndistorter::undistortLeftCallback, this);
    _left_pub_rect = _n.advertise<sensor_msgs::Image>(namespace_name + "/" + stereo_rig + "/left/image_rect", 1000);
    _right_sub = _n.subscribe(namespace_name + "/" + stereo_rig + "/right/image_raw", 1000, &StereoUndistorter::undistortRightCallback, this);
    _right_pub_rect = _n.advertise<sensor_msgs::Image>(namespace_name + "/" + stereo_rig + "/right/image_rect", 1000);
    _info_left_pub = _n.advertise<sensor_msgs::CameraInfo>(namespace_name + "/" + stereo_rig  + "/left/camera_info", 1000);
    _info_right_pub = _n.advertise<sensor_msgs::CameraInfo>(namespace_name + "/" + stereo_rig  + "/right/camera_info", 1000);

    cv::Mat d_left, d_right;
    d_left = cv::Mat::zeros(5, 1, CV_64F);
    d_right = cv::Mat::zeros(5, 1, CV_64F);

    cv::Mat k_left, k_right, trasf_left, trasf_right;
    getCameraInfo("cam0", config, k_left, d_left, trasf_left);
    getCameraInfo("cam1", config, k_right, d_right, trasf_right);

    std::cout << "Left camera intrinsics: " << k_left << std::endl;
    std::cout << "Left camera distortion: " << d_left << std::endl;
    std::cout << "Right camera intrinsics: " << k_right << std::endl;
    std::cout << "Right camera distortion: " << d_right << std::endl;
    std::cout << "Left camera transformation: " << trasf_left << std::endl;
    std::cout << "Right camera transformation: " << trasf_right << std::endl;

    cv::Mat r_left = trasf_left(cv::Rect(0, 0, 3, 3));
    cv::Mat r_right = trasf_right(cv::Rect(0, 0, 3, 3));
    cv::Mat t_left = trasf_left(cv::Rect(3, 0, 1, 3));
    cv::Mat t_right = trasf_right(cv::Rect(3, 0, 1, 3));
    cv::Size image_size = cv::Size(1280, 800);
    cv::Mat r1, r2, p1, p2, q;
    cv::Rect roi_left, roi_right;
    cv::stereoRectify(k_left, d_left, k_right, d_right, 
                    image_size, r_right, t_right, 
                    r1, r2, p1, p2, q, cv::CALIB_ZERO_DISPARITY, 
                    0, image_size, &roi_left, &roi_right);

    cv::initUndistortRectifyMap(k_left, d_left, r1, p1, 
                                image_size, CV_32FC1, _map_left_x, _map_left_y);
    cv::initUndistortRectifyMap(k_right, d_right, r2, p2,
                                image_size, CV_32FC1, _map_right_x, _map_right_y);

    setCameraInfo("cam0", config, stereo_rig, frame_id,
                  distortion_model, r1, p1, _cam_info_left);
    setCameraInfo("cam1", config, stereo_rig, frame_id,
                  distortion_model, r2, p2, _cam_info_right);

    return;
}

void StereoUndistorter::undistortLeftCallback(const sensor_msgs::ImageConstPtr& left_img)
{
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(*left_img, sensor_msgs::image_encodings::BGR8);
    cv::Mat img_l = cv_ptr->image;
    cv::Mat img_l_rect;
    _cam_info_left.header.stamp = left_img->header.stamp;
    cv::remap(img_l, img_l_rect, _map_left_x, _map_left_y, cv::INTER_LINEAR);
    sensor_msgs::ImagePtr msg_left_rect = cv_bridge::CvImage(left_img->header, sensor_msgs::image_encodings::BGR8, img_l_rect).toImageMsg();
    msg_left_rect->header.stamp = _cam_info_left.header.stamp;
    _left_pub_rect.publish(msg_left_rect);
    _info_left_pub.publish(_cam_info_left);

    return;
}

void StereoUndistorter::undistortRightCallback(const sensor_msgs::ImageConstPtr& right_img)
{
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(*right_img, sensor_msgs::image_encodings::BGR8);
    cv::Mat img_r = cv_ptr->image;
    cv::Mat img_r_rect;
    _cam_info_right.header.stamp = right_img->header.stamp;
    cv::remap(img_r, img_r_rect, _map_right_x, _map_right_y, cv::INTER_LINEAR);
    sensor_msgs::ImagePtr msg_right_rect = cv_bridge::CvImage(right_img->header, sensor_msgs::image_encodings::BGR8, img_r_rect).toImageMsg();
    msg_right_rect->header.stamp = _cam_info_right.header.stamp;
    _right_pub_rect.publish(msg_right_rect);
    _info_right_pub.publish(_cam_info_right);

    return;
}


void getCameraInfo(const std::string& cam_name, const YAML::Node& config, 
    cv::Mat& k_mat, cv::Mat& d_mat, cv::Mat& trasf_mat)
{
    YAML::Node cam_node = config[cam_name];

    std::vector<double> intrinsics = cam_node["intrinsics"].as<std::vector<double>>();
    k_mat = cv::Mat::zeros(3, 3, CV_64F);
    k_mat.at<double>(0, 0) = intrinsics[0];
    k_mat.at<double>(1, 1) = intrinsics[1];
    k_mat.at<double>(0, 2) = intrinsics[2];
    k_mat.at<double>(1, 2) = intrinsics[3];
    k_mat.at<double>(2, 2) = 1.0;

    std::vector<double> dist_params = cam_node["distortion_coeffs"].as<std::vector<double>>();
    d_mat = cv::Mat::zeros(dist_params.size(), 1, CV_64F);
    for (int i = 0; i < dist_params.size(); ++i)
        d_mat.at<double>(i, 0) = dist_params[i];

    trasf_mat = cv::Mat::eye(4, 4, CV_64F);
    if ( cam_node["T_cn_cnm1"] )
    {
        std::vector<std::vector<double>> T_cn_cnm1 = cam_node["T_cn_cnm1"].as<std::vector<std::vector<double>>>();
        trasf_mat = cv::Mat(4, 4, CV_64F);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                trasf_mat.at<double>(i, j) = T_cn_cnm1[i][j];
    }

    return;
}


void setCameraInfo(const std::string& cam_name, const YAML::Node& config,
                   const std::string& stereo_rig, const std::string& frame_id,
                   const std::string& distortion_model, 
                   const cv::Mat& r_mat, const cv::Mat& p_mat,
                   sensor_msgs::CameraInfo& cam_info)
{
    YAML::Node cam_node = config[cam_name]; 
    //std::string rig_camera = cam_name == "cam0" ? "left" : "right";
    std::string rig_camera = "left";

    cam_info.header.frame_id = stereo_rig + "/" + rig_camera + frame_id;
    cam_info.header.stamp = ros::Time::now();

    std::vector<int> resolution = cam_node["resolution"].as<std::vector<int>>();
    cam_info.width = resolution[0];
    cam_info.height = resolution[1];
    cam_info.distortion_model = distortion_model;

    for ( int i = 0; i < cam_info.D.size(); cam_info.D[i++] = 0.0 );

    for (int i = 0; i < 9; ++i)
    {
        cam_info.K[i] = p_mat.at<double>(i / 3, i % 3);
        cam_info.R[i] = r_mat.at<double>(i / 3, i % 3);
    }

    for (int i = 0; i < 12; ++i)
        cam_info.P[i] = p_mat.at<double>(i / 4, i % 4);


    cam_info.binning_x = 0;
    cam_info.binning_y = 0;
    cam_info.roi.do_rectify = false;

    return;
}
