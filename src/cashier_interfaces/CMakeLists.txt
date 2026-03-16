cmake_minimum_required(VERSION 3.8)
project(cashier_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Item.msg"
  "msg/Placement.msg"
  "srv/SrvDepthPosition.srv"
  "srv/ComputePackingPlan.srv"
  "action/VoiceSession.action"
  "action/ScanItems.action"
  "action/ExecutePacking.action"
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
