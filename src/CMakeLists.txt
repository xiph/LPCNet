# Files brought in from Codec2 that are mandatory for LPCNet.# Set visibility of files brought in from Codec2 so that they're not visible to 
# users (such as Codec2 itself).

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${lpcnet_SOURCE_DIR}
)

message(${lpcnet_SOURCE_DIR}/include)

set(lpcnet_freedv_srcs
    burg.c
    ceps_codebooks.c
    common.c
    dump_data.c
    freq.c
    kiss99.c
    kiss_fft.c
    lpcnet.c
    lpcnet_dec.c
    lpcnet_enc.c
    lpcnet_plc.c
    nnet.c
    nnet_data.c
    pitch.c
    plc_data.c
)

if(LPCNET_C_PROC_FLAGS)
foreach(LPCNET_FILE IN LISTS lpcnet_freedv_srcs)
    set_source_files_properties(${LPCNET_FILE} PROPERTIES COMPILE_FLAGS ${LPCNET_C_PROC_FLAGS})
endforeach()
endif(LPCNET_C_PROC_FLAGS)

add_library(lpcnetfreedv SHARED ${lpcnet_freedv_srcs} ${codec2_import_srcs})

set_target_properties(lpcnetfreedv PROPERTIES
    PUBLIC_HEADER lpcnet.h
	VERSION ${LPCNET_VERSION}
	SOVERSION ${LPCNET_SOVERSION}
) 
target_include_directories(lpcnetfreedv INTERFACE
    $<INSTALL_INTERFACE:include/lpcnet>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
install(TARGETS lpcnetfreedv EXPORT lpcnetfreedv-config
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lpcnet COMPONENT dev
)

install(EXPORT lpcnetfreedv-config
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lpcnetfreedv COMPONENT dev
)
# Export libfreedv target for import into codec2 from build tree.
export(TARGETS lpcnetfreedv
    FILE ${CMAKE_BINARY_DIR}/lpcnetfreedv.cmake
)