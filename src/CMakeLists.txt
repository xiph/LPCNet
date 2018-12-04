cmake_minimum_required(VERSION 3.1)

project(lpcnet)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(dump_data
	denoise.c 
	kiss_fft.c 
	pitch.c 
	celt_lpc.c 
)
target_compile_definitions(dump_data PRIVATE "TRAINING=1")
target_include_directories(dump_data PRIVATE ${CMAKE_CURRENT_SOURCE}/../include)
target_link_libraries(dump_data PRIVATE
	"m"
)


add_executable(test_lpcnet
	lpcnet.c
	nnet.c
	nnet_data.c
)
target_compile_definitions(test_lpcnet PRIVATE "mfma" "mavx2")
target_include_directories(test_lpcnet PRIVATE ${CMAKE_CURRENT_SOURCE}/../include)
target_link_libraries(test_lpcnet PRIVATE
	"m"
)

# gcc -DTRAINING=1 -Wall -W -O3 -g -I../include denoise.c kiss_fft.c pitch.c celt_lpc.c -o dump_data -lm
# gcc -mfma -msse3 -mavx2 -o test_lpcnet -g -O2 -Wall -W -Wextra lpcnet.c nnet.c nnet_data.c -lm

