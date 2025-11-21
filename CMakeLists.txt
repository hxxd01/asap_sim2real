cmake_minimum_required(VERSION 3.10)
project(rl_sar VERSION 3.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


option(SIMULATION "Enable simulation build" ON)
# option(KALMAN_FILTER "Enable state estimation build" ON)


# if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
# set(CMAKE_BUILD_TYPE Release)
# endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPER_BUILD_TYPE)
if(UPPER_BUILD_TYPE STREQUAL "DEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

add_definitions(-DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DPOLICY_DIR="${CMAKE_CURRENT_SOURCE_DIR}/policy")

find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(onnxruntime REQUIRED)

add_subdirectory(library/thirdparty/atom_sdk)

include_directories(
    include
    library/core/rl_sdk
    library/core/loop
    library/core/fsm
    library/core/utils
    library/core/logger
    library/core/matplotlibcpp
    policy
)

####################
###    rl_sdk    ###
####################

add_library(rl_sdk 
    library/core/rl_sdk/rl_sdk.cpp
    library/core/utils/math_rl.cpp
    )
target_link_libraries(rl_sdk PUBLIC
    TBB::tbb
    Python3::Python
    Python3::Module
    onnxruntime::onnxruntime
    atom_sdk::eigen3
    yaml
)

# ####################
# ###   rl_atom    ###
# ####################
add_executable(rl_real_atom
    src/rl_real_atom.cpp
    ${IDL_SOURCES}
)
target_link_libraries(rl_real_atom
    atom_sdk
    rl_sdk
    Threads::Threads
)

####################
###    rl_sim    ###
####################
if(SIMULATION)
    message(STATUS "[INFO] simulation robot control is enabled. Adding simulation library.")
    add_subdirectory(library/thirdparty/simulation)
    file(GLOB MUJOCO_SIMULATE_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/mujoco_simulate/**.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/joystick/joystick.cc
    )
    set(MUJOCO_SIMULATE_DEPENDENCIES
        mujoco::mujoco
        glfw
    )
    add_executable(rl_sim 
            src/rl_sim.cpp
            ${MUJOCO_SIMULATE_SRC}
    )
    target_include_directories(rl_sim PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/mujoco_simulate
        ${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/joystick
    )

    target_link_libraries(rl_sim 
        rl_sdk
        yaml
        Threads::Threads
        ${MUJOCO_SIMULATE_DEPENDENCIES}
    )

endif()