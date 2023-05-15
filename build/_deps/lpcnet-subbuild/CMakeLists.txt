# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.26.3)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(lpcnet-populate NONE)



include(ExternalProject)
ExternalProject_Add(lpcnet-populate
                     "UPDATE_DISCONNECTED" "False" "DOWNLOAD_EXTRACT_TIMESTAMP" "FALSE" "URL" "http://rowetel.com/downloads/deep/lpcnet_191005_v1.0.tgz"
                    SOURCE_DIR          "C:/Users/slide/back/lpcnet-cmake/LPCNet/build/_deps/lpcnet-src"
                    BINARY_DIR          "C:/Users/slide/back/lpcnet-cmake/LPCNet/build/_deps/lpcnet-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


