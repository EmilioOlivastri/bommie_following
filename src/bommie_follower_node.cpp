#include "bommie_following/bommie_planner.hpp"

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    ros::init(argc, argv, "bommie_planner");
    
    BommiePlanner planner(argv[1]);
    std::cout << "Bommie planner initialized." << std::endl;
    ros::spin();

    return 0;
}