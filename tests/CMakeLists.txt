add_executable(test_linalg test_linalg.cpp)
target_link_libraries(test_linalg linalgcpp)

add_test(test_linalg test_linalg)
