add_executable(bridge_test bridge_test.cpp)
add_executable(record_joint record_joint.cpp)
add_executable(rpc_test rpc_test.cpp)
add_executable(rpc_server_test rpc_server_test.cpp)
set(EXAMPLE_TARGETS
    bridge_test
    record_joint
    rpc_test
    rpc_server_test)

foreach(target ${EXAMPLE_TARGETS})
    target_link_libraries(${target} PRIVATE atom_sdk)
endforeach()
