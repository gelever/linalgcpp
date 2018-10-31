add_executable(dense_basics dense_basics.cpp)
target_link_libraries(dense_basics linalgcpp)

add_executable(dense_lu dense_lu.cpp)
target_link_libraries(dense_lu linalgcpp)

add_executable(dense_qr dense_qr.cpp)
target_link_libraries(dense_qr linalgcpp)

add_test(example_dense_basics dense_basics)
add_test(example_dense_lu dense_lu)
add_test(example_dense_qr dense_qr)

function(copy_file filename)
    configure_file(
        "${PROJECT_SOURCE_DIR}/examples/dense/${filename}"
        "${PROJECT_BINARY_DIR}/examples/dense/${filename}" @ONLY)
endfunction(copy_file)

copy_file(data/mat2.txt)
copy_file(data/mat4.txt)
copy_file(data/mat4.2.txt)
