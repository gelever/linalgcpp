if (LINALGCPP_ENABLE_MPI)
    add_subdirectory(parlinalgcpp)
endif(LINALGCPP_ENABLE_MPI)

if (LINALGCPP_ENABLE_SUITESPARSE)
    add_subdirectory(sparsesolver)
endif(LINALGCPP_ENABLE_SUITESPARSE)

if (LINALGCPP_ENABLE_METIS)
    add_subdirectory(partition)
endif(LINALGCPP_ENABLE_METIS)

if (LINALGCPP_ENABLE_GRAPHTOOLS)
    add_subdirectory(graphtools)
endif(LINALGCPP_ENABLE_GRAPHTOOLS)
