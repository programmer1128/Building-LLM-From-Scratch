cmake_minimum_required(VERSION 3.24)
project(TokenizerTests LANGUAGES C CXX)

enable_testing()
find_package(GTest 1.14 CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost 1.81 CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(TOKENIZER_ENABLE_TSAN "Instrument native tests with ThreadSanitizer" OFF)
if(TOKENIZER_ENABLE_TSAN)
    add_compile_options(-fsanitize=thread -g)
    add_link_options(-fsanitize=thread)
endif()

# Source ownership is defined once. Tests link this library through public APIs.
add_library(tokenizer_engine
    ../exp/tokenizers/BPE_Tokenizer.cpp
    ../exp/tokenizers/SplitStateMachine.cpp
)

target_link_libraries(tokenizer_engine PUBLIC Boost::headers)

add_library(tokenizer_test_support cpp/support/test_support.cpp)
target_link_libraries(tokenizer_test_support PUBLIC tokenizer_engine Threads::Threads)

add_executable(tokenizer_tests
    cpp/tests/byte_round_trip_test.cpp
    cpp/tests/import_ranks_test.cpp
    cpp/tests/invalid_import_test.cpp
    cpp/tests/serialization_test.cpp
    cpp/tests/corrupt_artifact_test.cpp
    cpp/tests/special_tokens_test.cpp
    cpp/tests/invalid_id_test.cpp
    cpp/tests/split_partition_test.cpp
    cpp/tests/concurrency_test.cpp
)
target_link_libraries(tokenizer_tests PRIVATE tokenizer_test_support GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(tokenizer_tests)

# Process adapter used by pytest and optional benchmarks.
add_executable(tokenizer_adapter cpp/adapter/tokenizer_adapter.cpp)
target_link_libraries(tokenizer_adapter PRIVATE tokenizer_test_support ${CMAKE_DL_LIBS})


# DataLoader library and unit test suite
add_library(dataloader_engine
    ../DataLoaders/DataLoader.cpp
)
target_include_directories(dataloader_engine PUBLIC
    ../DataLoaders
)

add_executable(dataloader_tests
    DataLoaders_Testing/DataLoaderTest.cpp
)
target_link_libraries(dataloader_tests PRIVATE
    dataloader_engine
    GTest::gtest_main
)

gtest_discover_tests(dataloader_tests)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(tokenizer_profile SHARED tokenizer_profile.c)
    set_target_properties(tokenizer_profile PROPERTIES PREFIX "")
endif()
