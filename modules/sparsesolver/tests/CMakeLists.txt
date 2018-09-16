add_executable(test_sparsesolver test_sparsesolver.cpp)
target_link_libraries(test_sparsesolver sparsesolve)

set_target_properties(test_sparsesolver PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

add_test(test_sparsesolver test_sparsesolver)


