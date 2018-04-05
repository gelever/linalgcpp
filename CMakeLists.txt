###############################################################
# MIT License
# 
# Copyright (c) 2018 Pablo Arias
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
# OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
###############################################################

###############################################################
# Using https://github.com/pabloariasal/modern-cmake-sample
# Modified by gelever on April 1st, 2018 #
###############################################################

cmake_minimum_required(VERSION 3.5)
project(linalgcpp VERSION 1.0 LANGUAGES CXX)

find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

add_library(linalgcpp
    src/argparser.cpp
    src/blockoperator.cpp
    src/coomatrix.cpp
    src/densematrix.cpp
    src/eigensolver.cpp
    src/operator.cpp
    src/solvers.cpp
    src/sparsematrix.cpp
    src/timer.cpp
    src/vector.cpp
    src/vectorview.cpp
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)


#set_target_properties(linalgcpp PROPERTIES
#        CXX_STANDARD 11
#        CXX_STANDARD_REQUIRED YES
#        CXX_EXTENSIONS NO
#)

target_include_directories(linalgcpp
    PUBLIC
		$<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(linalgcpp
    PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
)

add_subdirectory(tests)

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/liblinalgcpp)

install(TARGETS linalgcpp
    EXPORT linalgcpp-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT linalgcpp-targets
    FILE linalgcppTargets.cmake
    NAMESPACE linalgcpp::
    DESTINATION ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/linalgcppConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/linalgcppConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/linalgcppConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/linalgcppConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/linalgcppConfigVersion.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
    )

export(EXPORT linalgcpp-targets
       FILE ${CMAKE_CURRENT_BINARY_DIR}/linalgcppTargets.cmake
       NAMESPACE linalgcpp::
       )
export(PACKAGE linalgcpp)
