# CMake generated Testfile for 
# Source directory: C:/Users/slide/back/lpcnet-cmake/LPCNet
# Build directory: C:/Users/slide/back/lpcnet-cmake/LPCNet/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(feature_extraction "sh" "-c" "PATH=\$PATH:C:/Users/slide/back/lpcnet-cmake/LPCNet/build/src:C:/Users/slide/back/lpcnet-cmake/LPCNet/build/unittest;
                        cd C:/Users/slide/back/lpcnet-cmake/LPCNet/unittest; 
                        pwd;
                        dump_data --test --c2pitch C:/Users/slide/back/lpcnet-cmake/LPCNet/wav/birch.wav birch.f32;
                        md5sum birch.f32;
                        md5sum birch_targ.f32;
                        diff32 --cont birch_targ.f32 birch.f32")
set_tests_properties(feature_extraction PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/slide/back/lpcnet-cmake/LPCNet/CMakeLists.txt;211;add_test;C:/Users/slide/back/lpcnet-cmake/LPCNet/CMakeLists.txt;0;")
add_test(nnet2f32 "sh" "-c" "cd C:/Users/slide/back/lpcnet-cmake/LPCNet/build; ./src/nnet2f32 t.f32")
set_tests_properties(nnet2f32 PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/slide/back/lpcnet-cmake/LPCNet/CMakeLists.txt;219;add_test;C:/Users/slide/back/lpcnet-cmake/LPCNet/CMakeLists.txt;0;")
add_test(SIMD_functions "sh" "-c" "cd C:/Users/slide/back/lpcnet-cmake/LPCNet/build; ./src/test_vec")
set_tests_properties(SIMD_functions PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/slide/back/lpcnet-cmake/LPCNet/CMakeLists.txt;221;add_test;C:/Users/slide/back/lpcnet-cmake/LPCNet/CMakeLists.txt;0;")
add_test(lpcnet_enc_dec "sh" "-c" "PATH=\$PATH:C:/Users/slide/back/lpcnet-cmake/LPCNet/build/src;
                        cd C:/Users/slide/back/lpcnet-cmake/LPCNet;
                        sox wav/wia.wav -t raw -r 16000 - | 
                        lpcnet_enc -s | 
                        lpcnet_dec -s > /dev/null")
set_tests_properties(lpcnet_enc_dec PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/slide/back/lpcnet-cmake/LPCNet/CMakeLists.txt;223;add_test;C:/Users/slide/back/lpcnet-cmake/LPCNet/CMakeLists.txt;0;")
subdirs("src")
