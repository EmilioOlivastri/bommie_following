#include <iostream>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>

// PCL Filtering and Downsampling
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h> 

// PCL - ROS Conversions
#include <pcl/conversions.h> 
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>

// PCL Plane Estimation
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

// Vision and countour detection
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>


class BommiePlanner
{
public:

    PCL_MAKE_ALIGNED_OPERATOR_NEW

    BommiePlanner(std::string topic);
    ~BommiePlanner();
    
    void wallFollowing(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);

private:

    void filterCloudCPU(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                        const std::string& field_name,
                        const float min_limit,
                        const float max_limit, 
                        pcl::PointCloud<pcl::PointXYZ>& cloud_filtered);

    void filterCloudCuda(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                         const std::string& field_name,
                         const float min_limit,
                         const float max_limit, 
                         pcl::PointCloud<pcl::PointXYZ>& cloud_filtered);

    inline void filterCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud, const std::string& field_name,
                            const float min_limit, const float max_limit, 
                            pcl::PointCloud<pcl::PointXYZ>& cloud_filtered)
    {
        return _cuda ? filterCloudCuda(cloud, field_name, min_limit, max_limit, cloud_filtered) : 
                       filterCloudCPU(cloud, field_name, min_limit, max_limit, cloud_filtered);
    }


    void downSampleCloudCuda(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                             pcl::PointCloud<pcl::PointXYZ>& cloud_filtered);

    void downSampleCloudCPU(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                            pcl::PointCloud<pcl::PointXYZ>& cloud_filtered);

    inline void downSampleCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                                pcl::PointCloud<pcl::PointXYZ>& cloud_filtered)
    {
        return _cuda ? downSampleCloudCuda(cloud, cloud_filtered) : 
                       downSampleCloudCPU(cloud, cloud_filtered);
    }

    void estimatePlane(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                       pcl::ModelCoefficients& coefficients,
                       pcl::PointIndices& inliers);

    bool estimateBestPlane(pcl::PointCloud<pcl::PointXYZ>& cloud,
                           pcl::PointCloud<pcl::PointXYZ>& plane,
                           Eigen::Vector3d& plane_normal);

    void projectCloud2Image(pcl::PointCloud<pcl::PointXYZ> cloud,
                            pcl::PointCloud<pcl::PointXYZ>& countour_cloud,
                            Eigen::Vector2d& closest_pt);


    void visualizeMarker(const Eigen::Vector2d& pt, const Eigen::Vector3d& color, 
                         const std_msgs::Header& header, const ros::Publisher& pub);

    ros::NodeHandle _n;
    ros::Subscriber _sub_pcl;
    ros::Publisher _pub_pcl;
    std::vector<ros::Publisher> _pub_planes;
    ros::Publisher _debug_closet;
    std::vector<ros::Publisher> _debug_closests;
    ros::Publisher _debug_new_goal;
    std::vector<ros::Publisher> _debug_new_goals;
    std_msgs::Header _header;
    
    std::string _topic;

    double _standoff_distance;
    double _eps;

    double _max_view_forward_distance;
    double _min_view_forward_distance;
    double _max_view_side_distance;
    double _min_view_side_distance;

    double _num_planes;
    double _length_plane;
    double _height_plane;
    double _angle_tolerance;

    double _leaf_size;
    double _min_fwd_distance;

    int _follow_wall;
    bool _debug;
    bool _cuda;

    Eigen::Vector3d _vertical_normal;
    std::map<std::string, int> _cuda_filter_map;

};