add_executable(test_linalg test_linalg.cpp)
target_link_libraries(test_linalg linalgcpp)

set_target_properties(test_linalg PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

add_test(test_linalg test_linalg)
