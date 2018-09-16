add_executable(test_partition test_partition.cpp)
target_link_libraries(test_partition partition)

target_compile_options(test_partition PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O2>)
target_compile_options(test_partition PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O2>)

set_target_properties(test_partition PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)


add_test(test_partition test_partition)
