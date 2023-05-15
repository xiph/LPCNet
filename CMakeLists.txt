# TODO: 1/ Will this work when cross compiling for Windows?  Another approach is to supply
#          flags manually on cmd line
#       2/ Should we standardise on just AVX?  As machine we run on
#          may be different to machine we build on
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum OS X deployment version")

cmake_minimum_required(VERSION 3.0)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
	cmake_policy(SET CMP0135 NEW)
endif()

project(LPCNet C)

option(DISABLE_CPU_OPTIMIZATION "Disable CPU optimization discovery." OFF)
option(AVX2 "Enable AVX2 CPU optimizations." OFF)
option(AVX "Enable AVX CPU optimizations." ON)
option(SSE "Enable SSE CPU optimizations." ON)
option(NEON "Enable NEON CPU optimizations for RPi." ON)

include(GNUInstallDirs)

include_directories(${CMAKE_SOURCE_DIR}/include)

mark_as_advanced(CLEAR
    CMAKE_INSTALL_BINDIR
    CMAKE_INSTALL_INCLUDEDIR
    CMAKE_INSTALL_LIBDIR
)

# Build universal ARM64 and x86_64 binaries on Mac.
if(BUILD_OSX_UNIVERSAL)
set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")
endif(BUILD_OSX_UNIVERSAL)

#
# Prevent in-source builds
# If an in-source build is attempted, you will still need to clean up a few
# files manually.
#
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds in ${CMAKE_BINARY_DIR} are not "
   "allowed, please remove ./CMakeCache.txt and ./CMakeFiles/, create a "
   "separate build directory and run cmake from there.")
endif("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")


# Set project version information. This should probably be done via external
# file at some point.
#
set(LPCNET_VERSION_MAJOR 0)
set(LPCNET_VERSION_MINOR 5)
# Set to patch level if needed, otherwise leave FALSE.
# Must be positive (non-zero) if set, since 0 == FALSE in CMake.
set(LPCNET_VERSION_PATCH FALSE)
set(LPCNET_VERSION "${LPCNET_VERSION_MAJOR}.${LPCNET_VERSION_MINOR}")
# Patch level version bumps should not change API/ABI.
set(LPCNET_SOVERSION "${LPCNET_VERSION_MAJOR}.${LPCNET_VERSION_MINOR}")
if(LPCNET_VERSION_PATCH)
    set(LPCNET_VERSION "${LPCNET_VERSION}.${LPCNET_VERSION_PATCH}")
endif()
message(STATUS "LPCNet version: ${LPCNET_VERSION}")

#
# Find the git hash if this is a working copy.
#
if(EXISTS ${CMAKE_SOURCE_DIR}/.git)
    find_package(Git QUIET)
    if(Git_FOUND)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" describe --always HEAD
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            RESULT_VARIABLE res
            OUTPUT_VARIABLE FREEDV_HASH
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        message(STATUS "freedv-gui current git hash: ${FREEDV_HASH}")
        add_definitions(-DGIT_HASH="${FREEDV_HASH}")
    else()
        message(WARNING "Git not found. Can not determine current commit hash.")
        add_definitions(-DGIT_HASH="Unknown")
    endif()
else()
        add_definitions(-DGIT_HASH="None")
endif()

# Set default flags
set(CMAKE_C_FLAGS "-Wall -W -Wextra -Wno-unused-function -O3 -g -I. -MD ${CMAKE_C_FLAGS} -DENABLE_ASSERTIONS")

# Arch specific stuff here
message(STATUS "Host system arch is: ${CMAKE_SYSTEM_PROCESSOR}")

# Detection of available CPU optimizations
if(NOT DISABLE_CPU_OPTIMIZATION)
    if(UNIX AND NOT APPLE)
        message(STATUS "Looking for available CPU optimizations on Linux/BSD system...")
        execute_process(COMMAND grep -c "avx2" /proc/cpuinfo
            OUTPUT_VARIABLE AVX2_PRESENT)
        execute_process(COMMAND grep -c "avx " /proc/cpuinfo
            OUTPUT_VARIABLE AVX_PRESENT)
        execute_process(COMMAND grep -c "sse4_1 " /proc/cpuinfo
            OUTPUT_VARIABLE SSE_PRESENT)
        execute_process(COMMAND grep -c "neon" /proc/cpuinfo
            OUTPUT_VARIABLE NEON_PRESENT)
    elseif(APPLE)
        if(BUILD_OSX_UNIVERSAL)
            # Presume AVX and SSE are enabled on the x86 side. (AVX2 is not guaranteed depending
            # on model.) The ARM side will auto-enable NEON optimizations by virtue of being aarch64.
            set(AVX_PRESENT TRUE)
            set(SSE_PRESENT TRUE)
            set(NEON_PRESENT TRUE)
        else()
            # Under OSX we need to look through a few sysctl entries to determine what our CPU supports.
            message(STATUS "Looking for available CPU optimizations on an OSX system...")
            execute_process(COMMAND sysctl -a COMMAND grep machdep.cpu.leaf7_features COMMAND grep -c AVX2
                OUTPUT_VARIABLE AVX2_PRESENT)
            execute_process(COMMAND sysctl -a COMMAND grep machdep.cpu.features COMMAND grep -c AVX
                OUTPUT_VARIABLE AVX_PRESENT)
            execute_process(COMMAND sysctl -a COMMAND grep machdep.cpu.features COMMAND grep -c SSE4.1
                OUTPUT_VARIABLE SSE_PRESENT)
                
            # Unlike with the above, NEON *is* guaranteed if on ARM as there were never any ARM32 Macs 
            # available. We don't need any specific compiler flags for this, though.
            set(NEON_PRESENT TRUE)
        endif(BUILD_OSX_UNIVERSAL)
    elseif(WIN32)
        message(STATUS "No detection capability on Windows, assuming AVX is available.")
        set(AVX_PRESENT TRUE)
    else()
        message(STATUS "System is not *nix, processor specific optimizations cannot be determined.")
        message("   You can try setting them manually, e.g.: -DAVX2=1 or -DAVX=1 or -DNEON=1")
        
        # Presume all optimizations are available as the user will need to set them themselves.
        set(AVX2_PRESENT TRUE)
        set(AVX_PRESENT TRUE)
        set(SSE_PRESENT TRUE)
        set(NEON_PRESENT TRUE)
    endif()
else()
    # Presume all optimizations are available as the user is likely setting them themselves
    # (e.g. cross-compiling)
    set(AVX2_PRESENT TRUE)
    set(AVX_PRESENT TRUE)
    set(SSE_PRESENT TRUE)
    set(NEON_PRESENT TRUE)
endif()

set(LPCNET_C_PROC_FLAGS "")

if(${AVX2} AND (${AVX2_PRESENT} OR ${AVX2_PRESENT} GREATER 0))
    message(STATUS "avx2 processor flags found or enabled.")
    set(LPCNET_C_PROC_FLAGS -mavx2 -mfma)
elseif(${AVX} AND (${AVX_PRESENT} OR ${AVX_PRESENT} GREATER 0))
# AVX2 machines will also match on AVX
    message(STATUS "avx processor flags found or enabled.")
    set(LPCNET_C_PROC_FLAGS -mavx)
elseif(${SSE} AND (${SSE_PRESENT} OR ${SSE_PRESENT} GREATER 0))
# AVX and AVX2 machines will also match on SSE
    message(STATUS "sse processor flags found or enabled.")
    set(LPCNET_C_PROC_FLAGS -msse4.1)
elseif(${NEON} AND (${NEON_PRESENT} OR ${NEON_PRESENT} GREATER 0))
    # RPi / ARM 32bit
    message(STATUS "neon processor flags found or enabled.")
    set(LPCNET_C_PROC_FLAGS -mfpu=neon -march=armv8-a -mtune=cortex-a53)
endif()

# grab latest NN model (or substitute your own)
set(LPCNET_ROOT http://rowetel.com/downloads/deep/)
set(LPCNET_FILE lpcnet_191005_v1.0.tgz)
set(LPCNET_URL ${LPCNET_ROOT}${LPCNET_FILE})

if(EXISTS ${CMAKE_BINARY_DIR}/${LPCNET_FILE})
    set(lpcnet_SOURCE_DIR ${CMAKE_BINARY_DIR}/src)
    file(MAKE_DIRECTORY ${lpcnet_SOURCE_DIR})
    execute_process(COMMAND tar -xzf ${CMAKE_BINARY_DIR}/${LPCNET_FILE} -C ${CMAKE_BINARY_DIR}/src)

# Work around not having the FetchContent module.
elseif(CMAKE_VERSION VERSION_LESS 3.11.4)
    set(lpcnet_SOURCE_DIR ${CMAKE_BINARY_DIR}/src)
    if(NOT EXISTS ${lpcnet_SOURCE_DIR})
        file(DOWNLOAD ${LPCNET_URL}
            ${CMAKE_BINARY_DIR}/${LPCNET_FILE}
            SHOW_PROGRESS
        )
        file(MAKE_DIRECTORY ${lpcnet_SOURCE_DIR})
        execute_process(COMMAND tar -xzf ${CMAKE_BINARY_DIR}/${LPCNET_FILE} -C ${CMAKE_BINARY_DIR}/src)
    endif()
else()
    include(FetchContent)
    FetchContent_Declare(
        lpcnet
        URL ${LPCNET_URL})
    FetchContent_GetProperties(lpcnet)
    if(NOT lpcnet_POPULATED)
        FetchContent_Populate(lpcnet)
    endif()
endif()

add_subdirectory(src)

# Ctests ----------------------------------------------------------------------

include(CTest)
enable_testing()

# some basic tests - unfortunately the core NN synthsis is hard to test reliably as we get
# different results on different runs due to numerical/precision issues
add_test(NAME feature_extraction
         COMMAND sh -c "PATH=$PATH:${CMAKE_CURRENT_BINARY_DIR}/src:${CMAKE_CURRENT_BINARY_DIR}/unittest;
                        cd ${CMAKE_CURRENT_SOURCE_DIR}/unittest; 
                        pwd;
                        dump_data --test --c2pitch ${CMAKE_CURRENT_SOURCE_DIR}/wav/birch.wav birch.f32;
                        md5sum birch.f32;
                        md5sum birch_targ.f32;
                        diff32 --cont birch_targ.f32 birch.f32")
add_test(NAME nnet2f32
         COMMAND sh -c "cd ${CMAKE_CURRENT_BINARY_DIR}; ./src/nnet2f32 t.f32")
add_test(NAME SIMD_functions
         COMMAND sh -c "cd ${CMAKE_CURRENT_BINARY_DIR}; ./src/test_vec")
add_test(NAME lpcnet_enc_dec
         COMMAND sh -c "PATH=$PATH:${CMAKE_CURRENT_BINARY_DIR}/src;
                        cd ${CMAKE_CURRENT_SOURCE_DIR};
                        sox wav/wia.wav -t raw -r 16000 - | 
                        lpcnet_enc -s | 
                        lpcnet_dec -s > /dev/null")

# Packaging
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Next-Generation Digital Voice for Two-Way Radio")
set(CPACK_PACKAGE_VENDOR "CMake")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYING")
set(CPACK_PACKAGE_VERSION_MAJOR ${LPCNET_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${LPCNET_VERSION_MINOR})
if(LPCNET_VERSION_PATCH)
    set(CPACK_PACKAGE_VERSION_PATCH ${LPCNET_VERSION_PATCH})
else()
    set(CPACK_PACKAGE_VERSION_PATCH 0)
endif()

# Return the date (yyyy-mm-dd)
macro(DATE RESULT)
  execute_process(COMMAND "date" "+%Y%m%d" OUTPUT_VARIABLE ${RESULT})
endmacro()
DATE(DATE_RESULT)
string(REGEX REPLACE "\n$" "" DATE_RESULT "${DATE_RESULT}")
message(STATUS "Compilation date = XX${DATE_RESULT}XX")

set(CPACK_PACKAGE_VERSION_PATCH "${CPACK_PACKAGE_VERSION_PATCH}-${DATE_RESULT}-${FREEDV_HASH}")

if(UNIX AND NOT APPLE)
    # Linux packaging
    SET(CPACK_GENERATOR "DEB")
    SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Mooneer Salem <mooneer@gmail.com>") #required
    SET(CPACK_DEBIAN_PACKAGE_DEPENDS "")
    SET(CPACK_DEB_COMPONENT_INSTALL ON)
    SET(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
    SET(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)
    SET(CPACK_DEBIAN_LIB_PACKAGE_NAME "LPCNet")
    include(CPack)
    cpack_add_component(lib REQUIRED)
    cpack_add_component(dev DEPENDS lib)
    cpack_add_component(tools DEPENDS lib)
endif(UNIX AND NOT APPLE)

