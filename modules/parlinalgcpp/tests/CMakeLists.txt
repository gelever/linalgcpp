add_executable(test_parlinalgcpp test_parlinalg.cpp)
target_link_libraries(test_parlinalgcpp parlinalgcpp)

set_target_properties(test_parlinalgcpp PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

configure_file(${CMAKE_CURRENT_LIST_DIR}/test_data/test_mat.bin
    ${CMAKE_CURRENT_BINARY_DIR}/test_data/test_mat.bin
            COPYONLY)

configure_file(${CMAKE_CURRENT_LIST_DIR}/test_data/test_part.bin
    ${CMAKE_CURRENT_BINARY_DIR}/test_data/test_part.bin
            COPYONLY)

add_test(test_parlinalgcpp mpirun -np 2 ./test_parlinalgcpp)
