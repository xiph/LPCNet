# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "C:/Users/slide/back/lpcnet-cmake/LPCNet/build/_deps/lpcnet-src"
  "C:/Users/slide/back/lpcnet-cmake/LPCNet/build/_deps/lpcnet-build"
  "C:/Users/slide/back/lpcnet-cmake/LPCNet/build/_deps/lpcnet-subbuild/lpcnet-populate-prefix"
  "C:/Users/slide/back/lpcnet-cmake/LPCNet/build/_deps/lpcnet-subbuild/lpcnet-populate-prefix/tmp"
  "C:/Users/slide/back/lpcnet-cmake/LPCNet/build/_deps/lpcnet-subbuild/lpcnet-populate-prefix/src/lpcnet-populate-stamp"
  "C:/Users/slide/back/lpcnet-cmake/LPCNet/build/_deps/lpcnet-subbuild/lpcnet-populate-prefix/src"
  "C:/Users/slide/back/lpcnet-cmake/LPCNet/build/_deps/lpcnet-subbuild/lpcnet-populate-prefix/src/lpcnet-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/slide/back/lpcnet-cmake/LPCNet/build/_deps/lpcnet-subbuild/lpcnet-populate-prefix/src/lpcnet-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/slide/back/lpcnet-cmake/LPCNet/build/_deps/lpcnet-subbuild/lpcnet-populate-prefix/src/lpcnet-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
