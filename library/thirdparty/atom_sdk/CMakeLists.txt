cmake_minimum_required(VERSION 3.16)
project(atom_sdk LANGUAGES CXX)

option(BUILD_EXAMPLES "Build examples" ON)

# 全局设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(GNUInstallDirs)

# # Put all binary files into /bin and libraries into /lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

find_package(Threads REQUIRED)

# 导入第三方库
add_subdirectory(thirdparty)

add_library(atom_sdk INTERFACE)

target_link_libraries(atom_sdk INTERFACE     
    "${CMAKE_CURRENT_LIST_DIR}/lib/libclient_rpc.a"
    "${CMAKE_CURRENT_LIST_DIR}/lib/libserver_rpc.a"
    "${CMAKE_CURRENT_LIST_DIR}/lib/libcommon.a"
    dds eigen3 nlohmann yaml Threads::Threads)
target_include_directories(atom_sdk INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)



if (BUILD_EXAMPLES)
    add_subdirectory(example)
endif ()


install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${ATOM_SDK_LIB}
    DESTINATION ${CMAKE_INSTALL_LIBDIR})

# install(FILES cmake/atom_sdkTargets.cmake
#     DESTINATION lib/cmake/atom_sdk)

# include(CMakePackageConfigHelpers)
# write_basic_package_version_file(
#     atom_sdkConfigVersion.cmake
#     VERSION "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}.${${PROJECT_NAME}_VERSION_PATCH}"
#     COMPATIBILITY ExactVersion)

# configure_file(cmake/atom_sdkConfig.cmake.in atom_sdkConfig.cmake @ONLY)
# install(FILES "${CMAKE_CURRENT_BINARY_DIR}/atom_sdkConfig.cmake"
#     "${CMAKE_CURRENT_BINARY_DIR}/atom_sdkConfigVersion.cmake"
#     DESTINATION lib/cmake/atom_sdk)
