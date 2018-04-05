# Apply flags to all test targets
set(GLOBAL PROPERTY CXX_STANDARD 11)
set(GLOBAL PROPERTY CXX_STANDARD_REQUIRED YES)
set(GLOBAL PROPERTY CXX_EXTENSIONS NO)

add_executable(tests test.cpp)
target_link_libraries(tests linalgcpp)
