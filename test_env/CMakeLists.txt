cmake_minimum_required(VERSION 3.5)
project(test_env)

find_package(ament_cmake REQUIRED)

# Install scripts to lib/test_env so they are found by 'ros2 run'
install(PROGRAMS scripts/auto_coverage_mapper.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME auto_coverage_mapper
)
install(PROGRAMS scripts/concentration_mapper.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME concentration_mapper
)

# Install other directories to share/test_env
install(DIRECTORY launch navigation_config ros_params scenarios
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
