# 查找系统依赖
find_package(Threads REQUIRED)

############################# DDS库配置 ##############################
add_library(dds INTERFACE)

# 导入ddsc库
add_library(ddsc SHARED IMPORTED GLOBAL)
set_target_properties(ddsc PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/DDS/lib/libddsc.so"
  IMPORTED_NO_SONAME TRUE
)

# 导入cycloneddsidl库
add_library(cycloneddsidl SHARED IMPORTED GLOBAL)
set_target_properties(cycloneddsidl PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/DDS/lib/libcycloneddsidl.so"
  IMPORTED_NO_SONAME TRUE
)

# 导入安全相关库
add_library(dds_security_ac SHARED IMPORTED GLOBAL)
set_target_properties(dds_security_ac PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/DDS/lib/libdds_security_ac.so"
)

add_library(dds_security_auth SHARED IMPORTED GLOBAL)
set_target_properties(dds_security_auth PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/DDS/lib/libdds_security_auth.so"
)

add_library(dds_security_crypto SHARED IMPORTED GLOBAL)
set_target_properties(dds_security_crypto PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/DDS/lib/libdds_security_crypto.so"
)

# 汇总DDS目标（添加线程依赖）
target_link_libraries(dds INTERFACE
  ddsc 
  cycloneddsidl 
  dds_security_ac 
  dds_security_auth 
  dds_security_crypto
  Threads::Threads  # 显式链接线程库
)

# 包含目录（修正路径）
target_include_directories(dds INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/DDS/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>  # 安装后路径
)

# 安装配置（修正源路径，避免GLOB风险）
set(DDS_LIBS
  "${CMAKE_CURRENT_LIST_DIR}/DDS/lib/libddsc.so"
  "${CMAKE_CURRENT_LIST_DIR}/DDS/lib/libcycloneddsidl.so"
  "${CMAKE_CURRENT_LIST_DIR}/DDS/lib/libdds_security_ac.so"
  "${CMAKE_CURRENT_LIST_DIR}/DDS/lib/libdds_security_auth.so"
  "${CMAKE_CURRENT_LIST_DIR}/DDS/lib/libdds_security_crypto.so"
)
install(FILES ${DDS_LIBS} DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/DDS/include/"  # 修正源目录
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

message(STATUS "Configured DDS libraries")


############################# Eigen3 ##############################
add_library(eigen3 INTERFACE)
add_library(atom_sdk::eigen3 ALIAS eigen3)

# 包含目录（修正INSTALL_INTERFACE）
target_include_directories(eigen3 INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/eigen3>  # 假设eigen3/Eigen/...在此目录
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>  # 安装到include目录
)

# 安装头文件（保持目录结构，添加DESTINATION）
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/eigen3/"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}  # 添加DESTINATION参数
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")  # 只安装头文件


message(STATUS "Configured Eigen3 (header-only)")


############################# nlohmann ##############################
add_library(nlohmann INTERFACE)

target_include_directories(nlohmann INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>  # 假设nlohmann/json.hpp在此
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>  # 安装到include/nlohmann
)

# 安装头文件（精确指定文件）
install(FILES "${CMAKE_CURRENT_LIST_DIR}/nlohmann/json.hpp"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")  # 修正路径

message(STATUS "Configured nlohmann/json (header-only)")


############################# yaml-cpp ##############################
add_library(yaml STATIC IMPORTED GLOBAL)

set_target_properties(yaml PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/yaml/lib/libyaml-cpp.a"
)

target_include_directories(yaml INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/yaml/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# 安装库和头文件（修正源目录，添加DESTINATION）
install(FILES "${CMAKE_CURRENT_LIST_DIR}/yaml/lib/libyaml-cpp.a"
        DESTINATION ${CMAKE_INSTALL_LIBDIR})  # 添加DESTINATION
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/yaml/include/"  # 修正源目录
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})  # 添加DESTINATION

message(STATUS "Configured yaml-cpp library")