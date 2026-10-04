# CMake generated Testfile for 
# Source directory: /home/thor/Desktop/voice agent/whisper.cpp/tests
# Build directory: /home/thor/dbot_work/whisper_build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test-whisper-cli-tiny "/home/thor/dbot_work/whisper_build/bin/whisper-cli" "-m" "/home/thor/Desktop/voice agent/whisper.cpp/models/for-tests-ggml-tiny.bin" "-l" "fr" "-f" "/home/thor/Desktop/voice agent/whisper.cpp/samples/jfk.wav")
set_tests_properties(test-whisper-cli-tiny PROPERTIES  LABELS "tiny;gh" _BACKTRACE_TRIPLES "/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;19;add_test;/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;0;")
add_test(test-whisper-cli-tiny.en "/home/thor/dbot_work/whisper_build/bin/whisper-cli" "-m" "/home/thor/Desktop/voice agent/whisper.cpp/models/for-tests-ggml-tiny.en.bin" "-f" "/home/thor/Desktop/voice agent/whisper.cpp/samples/jfk.wav")
set_tests_properties(test-whisper-cli-tiny.en PROPERTIES  LABELS "tiny;en" _BACKTRACE_TRIPLES "/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;26;add_test;/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;0;")
add_test(test-whisper-cli-base "/home/thor/dbot_work/whisper_build/bin/whisper-cli" "-m" "/home/thor/Desktop/voice agent/whisper.cpp/models/for-tests-ggml-base.bin" "-l" "fr" "-f" "/home/thor/Desktop/voice agent/whisper.cpp/samples/jfk.wav")
set_tests_properties(test-whisper-cli-base PROPERTIES  LABELS "base" _BACKTRACE_TRIPLES "/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;33;add_test;/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;0;")
add_test(test-whisper-cli-base.en "/home/thor/dbot_work/whisper_build/bin/whisper-cli" "-m" "/home/thor/Desktop/voice agent/whisper.cpp/models/for-tests-ggml-base.en.bin" "-f" "/home/thor/Desktop/voice agent/whisper.cpp/samples/jfk.wav")
set_tests_properties(test-whisper-cli-base.en PROPERTIES  LABELS "base;en" _BACKTRACE_TRIPLES "/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;40;add_test;/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;0;")
add_test(test-whisper-cli-small "/home/thor/dbot_work/whisper_build/bin/whisper-cli" "-m" "/home/thor/Desktop/voice agent/whisper.cpp/models/for-tests-ggml-small.bin" "-l" "fr" "-f" "/home/thor/Desktop/voice agent/whisper.cpp/samples/jfk.wav")
set_tests_properties(test-whisper-cli-small PROPERTIES  LABELS "small" _BACKTRACE_TRIPLES "/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;47;add_test;/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;0;")
add_test(test-whisper-cli-small.en "/home/thor/dbot_work/whisper_build/bin/whisper-cli" "-m" "/home/thor/Desktop/voice agent/whisper.cpp/models/for-tests-ggml-small.en.bin" "-f" "/home/thor/Desktop/voice agent/whisper.cpp/samples/jfk.wav")
set_tests_properties(test-whisper-cli-small.en PROPERTIES  LABELS "small;en" _BACKTRACE_TRIPLES "/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;54;add_test;/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;0;")
add_test(test-whisper-cli-medium "/home/thor/dbot_work/whisper_build/bin/whisper-cli" "-m" "/home/thor/Desktop/voice agent/whisper.cpp/models/for-tests-ggml-medium.bin" "-l" "fr" "-f" "/home/thor/Desktop/voice agent/whisper.cpp/samples/jfk.wav")
set_tests_properties(test-whisper-cli-medium PROPERTIES  LABELS "medium" _BACKTRACE_TRIPLES "/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;61;add_test;/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;0;")
add_test(test-whisper-cli-medium.en "/home/thor/dbot_work/whisper_build/bin/whisper-cli" "-m" "/home/thor/Desktop/voice agent/whisper.cpp/models/for-tests-ggml-medium.en.bin" "-f" "/home/thor/Desktop/voice agent/whisper.cpp/samples/jfk.wav")
set_tests_properties(test-whisper-cli-medium.en PROPERTIES  LABELS "medium;en" _BACKTRACE_TRIPLES "/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;68;add_test;/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;0;")
add_test(test-whisper-cli-large "/home/thor/dbot_work/whisper_build/bin/whisper-cli" "-m" "/home/thor/Desktop/voice agent/whisper.cpp/models/for-tests-ggml-large.bin" "-f" "/home/thor/Desktop/voice agent/whisper.cpp/samples/jfk.wav")
set_tests_properties(test-whisper-cli-large PROPERTIES  LABELS "large" _BACKTRACE_TRIPLES "/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;75;add_test;/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;0;")
add_test(test-vad "/home/thor/dbot_work/whisper_build/bin/test-vad")
set_tests_properties(test-vad PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;99;add_test;/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;0;")
add_test(test-vad-full "/home/thor/dbot_work/whisper_build/bin/test-vad-full")
set_tests_properties(test-vad-full PROPERTIES  LABELS "base;en" _BACKTRACE_TRIPLES "/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;111;add_test;/home/thor/Desktop/voice agent/whisper.cpp/tests/CMakeLists.txt;0;")
