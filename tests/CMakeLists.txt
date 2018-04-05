# Apply flags to all test targets
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

add_executable(tests test.cpp)
target_link_libraries(tests linalgcpp)
