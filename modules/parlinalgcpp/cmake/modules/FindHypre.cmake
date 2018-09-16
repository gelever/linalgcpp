if(NOT Hypre_INCLUDE_DIR)
    find_path(Hypre_INCLUDE_DIR HYPRE.h
        HINTS ${Hypre_INC_DIR} ENV Hypre_INCLUDE_DIR ${Hypre_DIR} ENV Hypre_DIR
        PATH_SUFFIXES include
        DOC "Directory where the Hypre header files are located"
        )
endif()

if(NOT Hypre_LIBRARY)
    find_library(Hypre_LIBRARY
        NAMES HYPRE
        HINTS ${Hypre_LIB_DIR} ENV Hypre_LIB_DIR ${Hypre_DIR} ENV Hypre_DIR
        PATH_SUFFIXES lib
        DOC "Directory where the Hypre library is located"
  )
endif()

mark_as_advanced(HYPRE_INCLUDE_DIR HYPRE_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Hypre
    REQUIRED_VARS Hypre_LIBRARY Hypre_INCLUDE_DIR)

if (Hypre_FOUND AND NOT TARGET Hypre::Hypre)
    add_library(Hypre::Hypre INTERFACE IMPORTED)
    set_target_properties(Hypre::Hypre PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Hypre_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${Hypre_LIBRARY}"
        )
endif()
