add_executable(test_mis test_mis.cpp)
target_link_libraries(test_mis graphtools)

add_executable(test_parpart test_parpart.cpp)
target_link_libraries(test_parpart graphtools)

add_test(test_mis test_mis)
add_test(test_parpart test_parpart)
