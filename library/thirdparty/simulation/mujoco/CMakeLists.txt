cmake_minimum_required(VERSION 3.5)

# 设置 mujoco 路径
set(MUJOCO_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MUJOCO_INCLUDE_DIR ${MUJOCO_ROOT_DIR}/include)
set(MUJOCO_LIB_DIR ${MUJOCO_ROOT_DIR}/lib)

# 检查文件是否存在
if(NOT EXISTS ${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h)
    message(WARNING "MuJoCo header not found at: ${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h")
endif()

if(NOT EXISTS ${MUJOCO_LIB_DIR}/libmujoco.so)
    message(WARNING "MuJoCo library not found at: ${MUJOCO_LIB_DIR}/libmujoco.so")
endif()

# 创建导入的共享库目标
add_library(mujoco_shared SHARED IMPORTED GLOBAL)  # 添加 GLOBAL 关键字
set_target_properties(mujoco_shared PROPERTIES
    IMPORTED_LOCATION ${MUJOCO_LIB_DIR}/libmujoco.so
    INTERFACE_INCLUDE_DIRECTORIES ${MUJOCO_INCLUDE_DIR}
)

# 创建别名目标（遵循官方命名约定）
add_library(mujoco::mujoco ALIAS mujoco_shared)

# 导出变量供父项目使用
set(MUJOCO_INCLUDE_DIR ${MUJOCO_INCLUDE_DIR} PARENT_SCOPE)
set(MUJOCO_LIBRARY mujoco_shared PARENT_SCOPE)  # 导出实际目标名
set(MUJOCO_FOUND TRUE PARENT_SCOPE)

# 打印状态信息
message(STATUS "Found mujoco:")
message(STATUS "  Include dir: ${MUJOCO_INCLUDE_DIR}")
message(STATUS "  Library: ${MUJOCO_LIB_DIR}/libmujoco.so")