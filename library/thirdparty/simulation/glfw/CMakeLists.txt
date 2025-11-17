# 先尝试查找系统安装的 GLFW
find_package(glfw3 QUIET)
if(glfw3_FOUND)
    message(STATUS "Using system GLFW")
    # 创建别名以保持兼容性（glfw 而不是 glfw3）
    if(NOT TARGET glfw)
        add_library(glfw INTERFACE IMPORTED)
        set_target_properties(glfw PROPERTIES
            INTERFACE_LINK_LIBRARIES "glfw::glfw"
        )
    endif()
else()
    # 如果系统没有 GLFW，使用 FetchContent 下载
    message(STATUS "System GLFW not found, downloading via FetchContent")
    include(FetchContent)
    
    # 声明 GLFW 项目（使用 GitHub 仓库）
    FetchContent_Declare(
      glfw
      GIT_REPOSITORY https://github.com/glfw/glfw.git
      GIT_TAG        3.3.8  # 使用稳定版本而不是 master
    )
    
    # 设置 GLFW 构建选项（关闭不需要的部分）
    set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Build GLFW examples")
    set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "Build GLFW tests")
    set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "Build GLFW documentation")
    set(GLFW_INSTALL OFF CACHE INTERNAL "禁用安装")
    
    # 下载并构建 GLFW
    FetchContent_MakeAvailable(glfw)
    message(STATUS "GLFW has been successfully included via FetchContent.")
endif()