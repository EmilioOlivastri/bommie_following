#include "bommie_following/stereo_undistorter.hpp"


// The main function
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << "<stereo yaml>" << std::endl;
        return EXIT_FAILURE;
    }

    // Initialise ROS
    std::string calibration_file = argv[1];
    std::cout << "Loading stereo calibration file: " << calibration_file << std::endl;
    YAML::Node config = YAML::LoadFile(calibration_file);

    std::string node_name = ros::this_node::getName();
    ros::init(argc, argv, node_name.c_str());

    StereoUndistorter stereo_undistorter(config);
    ros::spin();

    return EXIT_SUCCESS;
}
