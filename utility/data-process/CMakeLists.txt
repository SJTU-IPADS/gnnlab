cmake_minimum_required(VERSION 3.14)
project(samgraph_utility CXX)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -g -fopenmp")

include_directories(.)
include_directories(${CMAKE_SOURCE_DIR}/../../3rdparty/CLI11/include)

set(COMMON_SOURCE
    ${CMAKE_SOURCE_DIR}/common/graph_loader.cc
    ${CMAKE_SOURCE_DIR}/common/options.cc
)

add_executable(
    comfriendster-generator
    ${CMAKE_SOURCE_DIR}/dataset/comfriendster_generator.cc
)

add_executable(
    get-degree
    ${CMAKE_SOURCE_DIR}/toolkit/degrees/get_degrees.cc
    ${COMMON_SOURCE}
)

add_executable(
    graph-perperty
    ${CMAKE_SOURCE_DIR}/toolkit/property/graph_property.cc
    ${COMMON_SOURCE}
)

add_executable(
    create-alias-table
    ${CMAKE_SOURCE_DIR}/toolkit/weight/create_alias_table.cc
    ${COMMON_SOURCE}
)

