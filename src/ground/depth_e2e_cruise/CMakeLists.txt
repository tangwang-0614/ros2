cmake_minimum_required(VERSION 3.0.2)
project(depth_e2e_cruise)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/main.py
  scripts/depth_e2e_core.py
  scripts/depth_e2e_node.py
  scripts/collect_data.py
  scripts/train.py
  scripts/evaluate.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
