add_executable(dense_ops dense_ops.cpp)
target_link_libraries(dense_ops linalgcpp)

function(copy_file filename)
    configure_file(
        "${PROJECT_SOURCE_DIR}/examples/dense/${filename}"
        "${PROJECT_BINARY_DIR}/examples/dense/${filename}" @ONLY)
endfunction(copy_file)

copy_file(data/mat2.txt)
copy_file(data/mat4.txt)
copy_file(data/mat4.2.txt)
