#include "bommie_following/bommie_planner.hpp"

#include <yaml-cpp/yaml.h>
#include <chrono>

#include "cudaPCL/cudaFilter.h"


#define LEFT 0
#define RIGHT 1

#define FAR_POINT 1000.0
#define MIN_POINTS_PER_PLANE 1000
#define INLIERS_PERC_PER_PLANE 0.25

#define COS_30 0.86602540378
#define CENTROID_SAME_ORI_EPS 0.4
#define CENTROID_DIFF_ORI_EPS 0.2
#define MAX_NUM_TRIALS 3

BommiePlanner::BommiePlanner(std::string config_file)
{
    YAML::Node config = YAML::LoadFile(config_file);

    _follow_wall = "left" == config["follow_wall"].as<std::string>() ? LEFT : RIGHT;
    _topic = config["topic_pcl"].as<std::string>();

    _standoff_distance = config["standoff_distance"].as<double>();
    _eps = config["eps_error"].as<double>(); // meters

    _max_view_forward_distance = config["max_view_forward"].as<double>();
    _min_view_forward_distance = config["min_view_forward"].as<double>();
    _max_view_side_distance = config["max_view_side"].as<double>();
    _min_view_side_distance = config["min_view_side"].as<double>();

    _num_planes = config["number_of_planes"].as<int>();
    _length_plane = config["plane_length"].as<double>();
    _height_plane = config["height_plane"].as<double>();
    _angle_tolerance = config["angle_tolerance"].as<double>();

    _leaf_size = config["leaf_size"].as<double>();
    _min_fwd_distance = config["min_view_forward"].as<double>();

    _debug = config["debug"].as<bool>();

    _sub_pcl = _n.subscribe(_topic, 1, &BommiePlanner::wallFollowing, this);
    _pub_pcl = _n.advertise<sensor_msgs::PointCloud2>("/contour", 1);
    _debug_closet = _n.advertise<visualization_msgs::Marker>("/closest_point", 1);
    _debug_new_goal = _n.advertise<visualization_msgs::Marker>("/new_goal", 1);

    std::string base_plane = "/plane";
    std::string base_closest = "/closest_point";
    std::string base_new_goal = "/new_goal";
    //for (int i = 0; i < _num_planes; ++i)
    for (int i = 0; i < 5; ++i)
    {
        _pub_planes.push_back(_n.advertise<sensor_msgs::PointCloud2>(base_plane + std::to_string(i), 1));
        _debug_closests.push_back(_n.advertise<visualization_msgs::Marker>(base_closest + std::to_string(i), 1));
        _debug_new_goals.push_back(_n.advertise<visualization_msgs::Marker>(base_new_goal + std::to_string(i), 1));
    }

    _cuda_filter_map.insert(std::pair<std::string, int>("x", 0));
    _cuda_filter_map.insert(std::pair<std::string, int>("y", 1));
    _cuda_filter_map.insert(std::pair<std::string, int>("z", 2));

    _vertical_normal = Eigen::Vector3d(1.0, 0.0, 0.0);
    _cuda = config["cuda"].as<bool>();
}

BommiePlanner::~BommiePlanner(){}


void BommiePlanner::wallFollowing(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{ 
    std::chrono::steady_clock::time_point real_begin = std::chrono::steady_clock::now();
    pcl::PointCloud<pcl::PointXYZ> cloud, cloud_filtered;

    pcl::fromROSMsg(*cloud_msg, cloud);
    double tot_points = static_cast<double>(cloud.size());
    _header = cloud_msg->header;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    //if (_debug ) pcl::io::savePCDFileASCII("./full_cloud.pcd", cloud);

    // Create a PassThrough filter to remove points outside the specified range
    filterCloud(cloud, "y", -_height_plane, _height_plane, cloud_filtered);
    filterCloud(cloud_filtered, "z", _min_view_forward_distance, _max_view_forward_distance, cloud);
    std::chrono::steady_clock::time_point filtering = std::chrono::steady_clock::now();

    // Downsampling the point cloud
    downSampleCloud(cloud, cloud_filtered);
    std::chrono::steady_clock::time_point donwsampling = std::chrono::steady_clock::now();

    //filterCloud(cloud_filtered, "x", -_max_view_side_distance, _max_view_side_distance, cloud);
    filterCloud(cloud_filtered, "x", -_max_view_side_distance, _max_view_side_distance, cloud);
    cloud_filtered = cloud;
    double filtered_points = static_cast<double>(cloud_filtered.size());
    std::chrono::steady_clock::time_point left_division = std::chrono::steady_clock::now();

    // Break point cloud into sections
    bool plane_found = false;
    std::vector<pcl::PointCloud<pcl::PointXYZ>> cloud_planes, debug_planes;
    std::vector<bool> plane_found_vec(_num_planes, false);
    std::vector<Eigen::Vector3d> normals(_num_planes, Eigen::Vector3d(0.0, 0.0, 0.0));
    cloud_planes.reserve(_num_planes);

    for (int i = 0; i < _num_planes; ++i)
    {
        double min_limit = _min_view_forward_distance + i * _length_plane;
        double max_limit = min_limit + _length_plane;
        pcl::PointCloud<pcl::PointXYZ> cloud_plane, best_plane;
        filterCloud(cloud_filtered, "z", min_limit, 
                     max_limit, cloud_plane);

        bool plane_found_local = estimateBestPlane(cloud_plane, best_plane, normals[i]);
        cloud_planes.emplace_back(plane_found_local ? best_plane : cloud_plane);
        plane_found_vec[i] = plane_found_local;
         
        // If there is at least one plane, it shoul be tru
        plane_found = plane_found || plane_found_vec[i];        
    }
    std::chrono::steady_clock::time_point plane_estimation = std::chrono::steady_clock::now();

    if (!plane_found)
    {
        std::cout << "No plane found!" << std::endl;
        return;
    }
        
    pcl::PointCloud<pcl::PointXYZ> countour;
    Eigen::Vector2d cls_pt(FAR_POINT, FAR_POINT);
    sensor_msgs::PointCloud2 contour_msg;
    projectCloud2Image(cloud_filtered, countour, cls_pt);
    std::chrono::steady_clock::time_point projection = std::chrono::steady_clock::now();

    pcl::toROSMsg(cloud_filtered, contour_msg);
    contour_msg.header = _header;
    _pub_pcl.publish(contour_msg);

    Eigen::Vector3d color_red(1.0, 0.0, 0.0);
    Eigen::Vector3d color_green(0.0, 1.0, 0.0);
    Eigen::Vector3d plane_zx(0.0, 1.0, 0.0);
    Eigen::Vector2d dir_2d(LEFT == _follow_wall ? 1.0 : -1.0, 0.0);
    for (int i = 0; i < _num_planes; ++i)
    {
        if (!plane_found_vec[i] ) continue;

        sensor_msgs::PointCloud2 plane_msg;
        pcl::PointCloud<pcl::PointXYZ> plane_countour;
        Eigen::Vector2d cls_pt_wall(FAR_POINT, FAR_POINT);    
        projectCloud2Image(cloud_planes[i], plane_countour, cls_pt_wall);
        
        // Keep constant distance from the wall
        Eigen::Vector2d goal(cls_pt_wall);

        // Project plane normal to z-x plane
        Eigen::Vector3d norm_prj_3d = normals[i] - (normals[i].dot(plane_zx) * plane_zx);
        norm_prj_3d.normalize(); 

        Eigen::Vector2d norm_prj(norm_prj_3d[0], norm_prj_3d[2]); 
        float cos_alpha = dir_2d.dot(norm_prj);
        norm_prj = cos_alpha < 0.0 ? -norm_prj : norm_prj; 
        norm_prj *= _standoff_distance;
        goal += norm_prj;

        //std::cout << "Plane normal 3D: " << normals[i].transpose() << std::endl;
        //std::cout << "Plane normal 2D: " << norm_prj.transpose() << std::endl;
        //std::cout << "Angle = " << cos_alpha << std::endl;

        _debug ? pcl::toROSMsg(cloud_planes[i], plane_msg) : pcl::toROSMsg(plane_countour, plane_msg);  
        plane_msg.header = _header;

        // Publish the filtered clouds and goals
        _pub_planes[i].publish(plane_msg);
        visualizeMarker(cls_pt_wall, color_red, contour_msg.header, _debug_closests[i]);
        visualizeMarker(goal, color_green, contour_msg.header, _debug_new_goals[i]);
    }
    
    // Keep constant distance
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::microseconds tot_time = std::chrono::duration_cast<std::chrono::microseconds>(end - real_begin);
    std::chrono::microseconds filtering_time = std::chrono::duration_cast<std::chrono::microseconds>(filtering - real_begin);
    std::chrono::microseconds downsampling_time = std::chrono::duration_cast<std::chrono::microseconds>(donwsampling - filtering);
    std::chrono::microseconds left_division_time = std::chrono::duration_cast<std::chrono::microseconds>(left_division - donwsampling);
    std::chrono::microseconds plane_estimation_time = std::chrono::duration_cast<std::chrono::microseconds>(plane_estimation - left_division);
    std::chrono::microseconds projection_time = std::chrono::duration_cast<std::chrono::microseconds>(projection - plane_estimation);
    std::cout << "Total Time (with publishing) " << tot_time.count() / 1000.0 << " [ms]" << std::endl;
    std::cout << "Filtering Time: " << filtering_time.count() / 1000.0 << " [ms]" << std::endl;
    std::cout << "Downsampling Time: " << downsampling_time.count() / 1000.0 << " [ms]" << std::endl;
    std::cout << "Left Division Time: " << left_division_time.count() / 1000.0 << " [ms]" << std::endl;
    std::cout << "Plane Estimation Time: " << plane_estimation_time.count() / 1000.0 << " [ms]" << std::endl;
    std::cout << "Projection Time: " << projection_time.count() / 1000.0 << " [ms]" << std::endl;
    //for (int i = 0; i < _num_planes; ++i)
    //    std::cout << "Plane " << i << ": " << plane_found_vec[i] << std::endl;

    std::cout << "#######################" << std::endl;
    

    return;
}


void BommiePlanner::filterCloudCPU(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                                   const std::string& field_name,
                                   const float min_limit,
                                   const float max_limit, 
                                   pcl::PointCloud<pcl::PointXYZ>& cloud_filtered)
{    
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud.makeShared());
    pass.setFilterFieldName(field_name);
    pass.setFilterLimits(min_limit, max_limit); 
    pass.filter(cloud_filtered);

    return;
}

void BommiePlanner::downSampleCloudCPU(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                                       pcl::PointCloud<pcl::PointXYZ>& cloud_filtered)
{
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud.makeShared());
    voxel_grid.setLeafSize(_leaf_size, _leaf_size, _leaf_size);
    voxel_grid.filter(cloud_filtered);

    return;
}

void BommiePlanner::filterCloudCuda(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                                    const std::string& field_name,
                                    const float min_limit,
                                    const float max_limit, 
                                    pcl::PointCloud<pcl::PointXYZ>& cloud_filtered)
{
    // Activating the CUDA stream between the host and device
    cudaStream_t stream = NULL;
    cudaStreamCreate ( &stream );
    
    // CPU Data Operations
    unsigned int num_points = cloud.width * cloud.height;
    float* input_cpu = (float *)cloud.points.data();

    // Allocate memory for the input data on the cuda device and copy the data
    float *input_cuda = NULL;
    cudaMallocManaged(&input_cuda, sizeof(float) * 4 * num_points, cudaMemAttachHost);
    cudaStreamAttachMemAsync (stream, input_cuda);
    cudaMemcpyAsync(input_cuda, input_cpu, sizeof(float) * 4 * num_points, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Allocate memory for the output data on the cuda device
    float *output_cuda = NULL;
    cudaMallocManaged(&output_cuda, sizeof(float) * 4 * num_points, cudaMemAttachHost);
    cudaStreamAttachMemAsync (stream, output_cuda);
    cudaStreamSynchronize(stream);

    // Creating the filter object
    cudaFilter filter(stream);
    FilterParam_t setP;
    FilterType_t type = PASSTHROUGH;
    setP.type = type;
    setP.dim = _cuda_filter_map[field_name];
    setP.upFilterLimits = max_limit;
    setP.downFilterLimits = min_limit;
    setP.limitsNegative = false;
    filter.set(setP);

    // Filtering the data
    unsigned int num_points_filtered = 0;
    cudaDeviceSynchronize();
    filter.filter(output_cuda, &num_points_filtered, input_cuda, num_points);
    cudaDeviceSynchronize();

    // Allocating memory for the output data on the host
    cloud_filtered.width = num_points_filtered; 
    cloud_filtered.height = 1;
    cloud_filtered.resize(cloud_filtered.width * cloud_filtered.height);
    float *output_cpu = (float *)cloud_filtered.points.data();
    memset(output_cpu, 0, sizeof(float) * 4 * num_points_filtered);

    // Copying the filtered data from the device to the host
    cudaMemcpyAsync(output_cpu, output_cuda, sizeof(float) * 4 * num_points_filtered, cudaMemcpyDeviceToHost, stream);
    cudaDeviceSynchronize();

    // Freeing the allocated memory
    cudaFree(input_cuda);
    cudaFree(output_cuda);
    cudaStreamDestroy(stream);

    return;
}

void BommiePlanner::downSampleCloudCuda(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                                        pcl::PointCloud<pcl::PointXYZ>& cloud_filtered)
{
    // Activating the CUDA stream between the host and device
    cudaStream_t stream = NULL;
    cudaStreamCreate ( &stream );

    // CPU Data Operations
    unsigned int num_points = cloud.width * cloud.height;
    float* input_cpu = (float *)cloud.points.data();

    // Allocate memory for the input data on the cuda device and copy the data
    float *input_cuda = NULL;
    cudaMallocManaged(&input_cuda, sizeof(float) * 4 * num_points, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, input_cuda);
    cudaMemcpyAsync(input_cuda, input_cpu, sizeof(float) * 4 * num_points, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Allocate memory for the output data on the cuda device
    float *output_cuda = NULL;
    cudaMallocManaged(&output_cuda, sizeof(float) * 4 * num_points, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, output_cuda);
    cudaStreamSynchronize(stream);

    // Creating the filter object
    cudaFilter filter(stream);
    FilterParam_t setP;
    FilterType_t type = VOXELGRID;
    setP.type = type;
    setP.voxelX = _leaf_size;
    setP.voxelY = _leaf_size;
    setP.voxelZ = _leaf_size;
    filter.set(setP);

    // Filtering the data
    unsigned int num_points_filtered = 0;
    cudaDeviceSynchronize();
    int status = filter.filter(output_cuda, &num_points_filtered, input_cuda, num_points);
    cudaDeviceSynchronize();
    
    // Allocating memory for the output data on the host
    cloud_filtered.width = num_points_filtered; 
    cloud_filtered.height = 1;
    cloud_filtered.resize(cloud_filtered.width * cloud_filtered.height);
    float *output_cpu = (float *)cloud_filtered.points.data();
    memset(output_cpu, 0, sizeof(float) * 4 * num_points_filtered);

    // Copying the filtered data from the device to the host
    cudaMemcpyAsync(output_cpu, output_cuda, sizeof(float) * 4 * num_points_filtered, cudaMemcpyDeviceToHost, stream);
    cudaDeviceSynchronize();

    // Freeing the allocated memory
    cudaFree(input_cuda);
    cudaFree(output_cuda);
    cudaStreamDestroy(stream);

    return;
}

void BommiePlanner::estimatePlane(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                                  pcl::ModelCoefficients& coefficients,
                                  pcl::PointIndices& inliers)
{
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.05);
    seg.setEpsAngle(_angle_tolerance * M_PI / 180.0);
    Eigen::Vector3f plane_normal(1.0f, 0.0f, 0.0f);
    seg.setAxis(plane_normal);
    seg.setInputCloud(cloud.makeShared());
    seg.segment(inliers, coefficients);

    return;
}

bool BommiePlanner::estimateBestPlane(pcl::PointCloud<pcl::PointXYZ>& cloud,
                                      pcl::PointCloud<pcl::PointXYZ>& plane,
                                      Eigen::Vector3d& plane_normal)
{
    pcl::PointCloud<pcl::PointXYZ> candidate_plane;
    Eigen::Vector3d candidate_normals;
    Eigen::Vector3d candidate_centroids;

    // Variables for estimation of the correct plane for wall following
    Eigen::Vector3d plane_zx(0.0, 1.0, 0.0);
    Eigen::Rotation2Dd rot_90(M_PI / 2.0);
    Eigen::Rotation2Dd rot_270(-M_PI / 2.0);
    for (int trial = 0; trial < MAX_NUM_TRIALS && cloud.size() > MIN_POINTS_PER_PLANE; ++trial)
    {
        pcl::PointIndices inliers;
        pcl::ModelCoefficients plane_coefs, coefs_right;
        estimatePlane(cloud, plane_coefs, inliers);

        int number_points = cloud.size();
        float inliers_perc = static_cast<float>(inliers.indices.size()) / static_cast<float>(cloud.size());
        bool success = inliers_perc > INLIERS_PERC_PER_PLANE;;

        if (!success) continue;

        // Extracting the points belonging to the plane
        pcl::PointCloud<pcl::PointXYZ> remaining_cloud, candidate_plane;
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud.makeShared());
        extract.setIndices(boost::make_shared<pcl::PointIndices>(inliers));
        extract.setNegative(false);
        extract.filter(candidate_plane);
        
        // Computing the plane normal and projecting it to the z-x plane
        Eigen::Vector3d candidate_normal(plane_coefs.values[0], plane_coefs.values[1], plane_coefs.values[2]);
        candidate_normal.normalize();
        Eigen::Vector3d norm_prj_3d = candidate_normal - (candidate_normal.dot(plane_zx) * plane_zx);
        norm_prj_3d.normalize(); 
        Eigen::Vector2d norm_prj(norm_prj_3d[0], norm_prj_3d[2]);

        // Computing the centroid of the plane
        Eigen::Vector2d centroid(candidate_plane[0].x, candidate_plane[0].z);
        for (size_t i = 1; i < candidate_plane.size(); ++i)
            centroid += Eigen::Vector2d(candidate_plane[i].x, candidate_plane[i].z);
        centroid /= static_cast<double>(candidate_plane.size());
        
        // Computing candidates goals to verify they are reachable and left following
        Eigen::Vector2d candidate_goal1 = centroid + norm_prj * _standoff_distance;
        Eigen::Vector2d candidate_goal2 = centroid - norm_prj * _standoff_distance;

        // Estimate line equation for the plane (perpendicular to the normal)
        Eigen::Vector2d grad1 = rot_90 * norm_prj;
        Eigen::Vector2d grad2 = rot_270 * norm_prj;
        Eigen::Vector2d pt1 = centroid + grad1 * _standoff_distance;
        Eigen::Vector2d pt2 = centroid + grad2 * _standoff_distance;
        Eigen::Vector2d grad = pt1.norm() < pt2.norm() ? grad1 : grad2;
        Eigen::Vector2d pt = pt1.norm() < pt2.norm() ? pt1 : pt2;
        double angle_plane = atan2(grad[1], grad[0]);
        double m_plane = tan(angle_plane); 
        double q_plane = pt[1] - m_plane * pt[0];

        // Estimate the line equation for candidate points (pass through the [0, 0])
        double m_c1 = atan2(candidate_goal1[1], candidate_goal1[0]);
        double m_c2 = atan2(candidate_goal2[1], candidate_goal2[0]);

        // Compute the intersection of the two lines
        Eigen::Vector2d intersec1, intersec2;
        intersec1[0] = q_plane / (m_c1 - m_plane);
        intersec1[1] = m_plane * intersec1[0] + q_plane;
        intersec2[0] = q_plane / (m_c2 - m_plane);
        intersec2[1] = m_plane * intersec2[0] + q_plane;

        // If the intersection is closer than the candidate points, 
        // it is not a valid candidate as we have to go trough the wall to reach it
        bool valid_candidate1 = intersec1.norm() > candidate_goal1.norm();
        bool valid_candidate2 = intersec2.norm() > candidate_goal2.norm();
        Eigen::Vector2d candidate_goal = valid_candidate1 ? candidate_goal1 : candidate_goal2;

        // Estimate possible goal point with orientation that the robot should have
        // for doing the wall following
        Eigen::Isometry2d goal_point = Eigen::Isometry2d::Identity();
        goal_point.linear() = Eigen::Rotation2Dd(angle_plane).toRotationMatrix();
        goal_point.translation() = candidate_goal;

        Eigen::Vector2d centroid_in_goal = goal_point.inverse() * centroid;
        //std::cout << "Centroid in goal: " << centroid_in_goal.transpose() << std::endl;
        bool found = centroid_in_goal[1] < -0.1;

        if (found && (valid_candidate1 || valid_candidate2))
        {
            plane = candidate_plane;
            plane_normal = candidate_normal;
            return true;
        }

        // Removing the points belonging to the plane 
        extract.setNegative(true);
        extract.filter(remaining_cloud);
        cloud = remaining_cloud;
    }

    return false;
}

void BommiePlanner::projectCloud2Image(pcl::PointCloud<pcl::PointXYZ> cloud,
                                       pcl::PointCloud<pcl::PointXYZ>& countour_cloud,
                                       Eigen::Vector2d& closest_pt)
{
    cv::Mat map = cv::Mat::zeros(720, 720, CV_8U);
    cv::Mat countour = map.clone();
    for (size_t i = 0; i < cloud.size(); cloud[++i].y = 0.0);

    int init_x = _follow_wall == LEFT ? map.cols - 1 : 0;
    int init_z = map.rows - 1;
    int step_x = _follow_wall == LEFT ? -1 : 1;

    double length_x = _max_view_side_distance;
    double length_z = _max_view_forward_distance;
    double res_x = length_x / map.cols;
    double res_z = length_z / map.rows;

    for (size_t i = 0; i < cloud.size(); ++i)
    {
        int x = static_cast<int>(init_x + cloud[i].x / res_x);
        int z = static_cast<int>(init_z - cloud[i].z / res_z);
        bool is_valid = (x >= 0 && x < map.cols && z >= 0 && z < map.rows);
        map.at<uchar>(z, x) = is_valid ? 255 : 0; 
    }

    // Keep countour

    // NOT THE OPTIMIZED VERSION, INITIAL VERSION
    // THERE IS NO NEED TO DO ALL THE POINTS, WE JUST NEED
    // TO FIND THE FIRST POINTS WHICH IS THE CLOSEST TO THE CAMERA
    // AND USE THAT FOR THE CONTROL OF THE ROBOT 
    // (maybe not only the first and closest, but also a neighbourhood of points)
    // The full countour is for visualization purposes
    int offset = map.rows; 
    countour_cloud.reserve(500);
    double left_most = -1000.0;
    for (int i = map.rows - 1; i > 0; --i)
    {
        bool first_found = false;
        for (int j = 0; j < map.cols && !first_found; ++j)
        {
            int real_j = init_x + j * step_x;
            first_found = map.at<uchar>(i, real_j) > 10;

            if (!first_found)
                continue;

            Eigen::Vector2d pt(step_x * j * res_x, (init_z - i) * res_z);
            double dist1 = pt.norm();
            left_most = pt[0] > left_most ? pt[0] : left_most;
            
            closest_pt = dist1 < closest_pt.norm() ? pt : closest_pt;
            countour.at<uchar>(i, real_j) = map.at<uchar>(i, real_j);
            countour_cloud.emplace_back(pcl::PointXYZ(pt[0], 0.0, pt[1]));
        }
    }

    //std::cout << "Closest point: " << closest_pt.transpose() << std::endl;
    closest_pt[0] = left_most;
    //std::cout << "Closest point w/ leftmost: " << closest_pt.transpose() << std::endl;
    //std::cout << "Closest point distance: " << closest_pt.norm() << std::endl;
    //std::cout << "Closest point angle: " << atan2(closest_pt[1], closest_pt[0]) * 180.0 / M_PI << std::endl;
    //std::cout << "Size of countour: " << countour_cloud.size() << std::endl;

    /**/
    bool debug = false;
    if (debug)
    {
        std::string window_name = _follow_wall == LEFT ? "Left" : "Right";
        cv::imshow(window_name, countour);
        cv::waitKey(1);
    }
    /**/

    return;
}


void BommiePlanner::visualizeMarker(const Eigen::Vector2d& pt,
                                    const Eigen::Vector3d& color, 
                                    const std_msgs::Header& header, 
                                    const ros::Publisher& pub)
{
    visualization_msgs::Marker marker;
    marker.header.frame_id = header.frame_id;
    marker.header.stamp = header.stamp;
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = pt[0];
    marker.pose.position.y = 0.0;
    marker.pose.position.z = pt[1];
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.r = color[0];
    marker.color.g = color[1];
    marker.color.b = color[2];
    marker.color.a = 1.0;

    pub.publish(marker);

    return;
}

