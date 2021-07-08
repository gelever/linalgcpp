add_executable(vector_ops vector_ops.cpp)
target_link_libraries(vector_ops linalgcpp)

add_test(example_vector_basics vector_ops)

function(copy_file filename)
    configure_file(
        "${PROJECT_SOURCE_DIR}/examples/vector/${filename}"
        "${PROJECT_BINARY_DIR}/examples/vector/${filename}" @ONLY)
endfunction(copy_file)

copy_file(data/vect2.txt)
copy_file(data/vect4.txt)
copy_file(data/vect4.2.txt)

