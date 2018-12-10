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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

find_package(Hypre REQUIRED)

add_library(parlinalgcpp
    src/hypre_ops.cpp
    src/parcommpkg.cpp
    src/parmatrix.cpp
    src/paroperator.cpp
    src/parsmoother.cpp
    src/parsolvers.cpp
    src/parutilities.cpp
    src/parvector.cpp
)

add_dependencies(parlinalgcpp linalgcpp)

target_compile_options(parlinalgcpp PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall -O2>)
target_compile_options(parlinalgcpp PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wall -O2>)

target_link_libraries(parlinalgcpp
    PUBLIC Hypre::Hypre linalgcpp
)

target_include_directories(parlinalgcpp
    PUBLIC
		$<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/libparlinalgcpp)

install(TARGETS parlinalgcpp
    EXPORT parlinalgcpp-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT parlinalgcpp-targets
    FILE parlinalgcppTargets.cmake
    NAMESPACE linalgcpp::
    DESTINATION ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/parlinalgcppConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/parlinalgcppConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/parlinalgcppConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/parlinalgcppConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/parlinalgcppConfigVersion.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/modules/FindHypre.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/FindHypre.cmake
            COPYONLY)

export(EXPORT parlinalgcpp-targets FILE ${CMAKE_CURRENT_BINARY_DIR}/parlinalgcppTargets.cmake NAMESPACE linalgcpp::)

export(PACKAGE parlinalgcpp)

add_subdirectory(tests)
