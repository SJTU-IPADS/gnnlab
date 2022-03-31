cmake_minimum_required(VERSION 3.14)
project(samgraph_test CXX CUDA)

# GoogleTest requires at least C++11
set(CMAKE_CXX_STANDARD 11)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 

enable_testing()

add_executable(
  samgraph_test
  cuda_class_test.cu
  device_query_test.cc
  memory_race_test.cu
  memcpy_test.cc
)

target_link_libraries(
  samgraph_test
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(samgraph_test)
