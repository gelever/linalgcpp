add_executable(sparse_basics sparse_basics.cpp)
target_link_libraries(sparse_basics linalgcpp)

add_test(example_sparse_basics sparse_basics)

function(copy_file filename)
    configure_file(
        "${PROJECT_SOURCE_DIR}/examples/sparse/${filename}"
        "${PROJECT_BINARY_DIR}/examples/sparse/${filename}" @ONLY)
endfunction(copy_file)

copy_file(data/internet.coo)
