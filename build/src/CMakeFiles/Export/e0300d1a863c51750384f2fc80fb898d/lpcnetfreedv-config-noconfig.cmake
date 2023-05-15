#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "lpcnetfreedv" for configuration ""
set_property(TARGET lpcnetfreedv APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(lpcnetfreedv PROPERTIES
  IMPORTED_IMPLIB_NOCONFIG "${_IMPORT_PREFIX}/lib/liblpcnetfreedv.dll.a"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/liblpcnetfreedv.dll"
  )

list(APPEND _cmake_import_check_targets lpcnetfreedv )
list(APPEND _cmake_import_check_files_for_lpcnetfreedv "${_IMPORT_PREFIX}/lib/liblpcnetfreedv.dll.a" "${_IMPORT_PREFIX}/bin/liblpcnetfreedv.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
