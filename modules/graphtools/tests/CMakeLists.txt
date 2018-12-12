
add_executable(test_graphtools test_graphtools.cpp)
target_link_libraries(test_graphtools graphtools)

add_test(test_graphtools test_graphtools)
