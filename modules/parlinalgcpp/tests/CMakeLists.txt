add_executable(test_parlinalgcpp test_parlinalg.cpp)
target_link_libraries(test_parlinalgcpp parlinalgcpp)

configure_file(${CMAKE_CURRENT_LIST_DIR}/test_data/test_mat.bin
    ${CMAKE_CURRENT_BINARY_DIR}/test_data/test_mat.bin
            COPYONLY)

configure_file(${CMAKE_CURRENT_LIST_DIR}/test_data/test_part.bin
    ${CMAKE_CURRENT_BINARY_DIR}/test_data/test_part.bin
            COPYONLY)

add_test(test_parlinalgcpp mpirun -np 2 ./test_parlinalgcpp)
