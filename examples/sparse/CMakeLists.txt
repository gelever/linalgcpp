add_executable(sparse_matvec sparse_matvec.cpp)
target_link_libraries(sparse_matvec linalgcpp)

function(copy_file filename)
    configure_file(
        "${PROJECT_SOURCE_DIR}/examples/sparse/${filename}"
        "${PROJECT_BINARY_DIR}/examples/sparse/${filename}" @ONLY)
endfunction(copy_file)

copy_file(data/internet.coo)
