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
    comfriendster-csr-generator
    ${CMAKE_SOURCE_DIR}/dataset/comfriendster_csr_generator.cc
)

add_executable(
    comfriendster-coo-generator
    ${CMAKE_SOURCE_DIR}/dataset/comfriendster_coo_generator.cc
)

add_executable(
    cache-by-degree
    ${CMAKE_SOURCE_DIR}/toolkit/cache/cache_by_degree.cc
    ${COMMON_SOURCE}
)

add_executable(
    cache-by-heuristic
    ${CMAKE_SOURCE_DIR}/toolkit/cache/cache_by_heuristic.cc
    ${COMMON_SOURCE}
)

add_executable(
    cache-by-degree-hop
    ${CMAKE_SOURCE_DIR}/toolkit/cache/cache_by_degree_hop.cc
    ${COMMON_SOURCE}
)
add_executable(
    cache-by-fake-optimal
    ${CMAKE_SOURCE_DIR}/toolkit/cache/cache_by_fake_optimal.cc
    ${COMMON_SOURCE}
)

add_executable(
    degree-info
    ${CMAKE_SOURCE_DIR}/toolkit/degree/degree_info.cc
    ${COMMON_SOURCE}
)

add_executable(
    32to64
    ${CMAKE_SOURCE_DIR}/toolkit/generator/32to64.cc
    ${COMMON_SOURCE}
)

add_executable(
    nodeset-generator
    ${CMAKE_SOURCE_DIR}/toolkit/generator/nodeset_generator.cc
    ${COMMON_SOURCE}
)

add_executable(
    coo-to-dataset
    ${CMAKE_SOURCE_DIR}/toolkit/generator/coo_to_dataset.cc
    ${COMMON_SOURCE}
)

add_executable(
    load-mem
    ${CMAKE_SOURCE_DIR}/toolkit/load/load_mem.cc
    ${COMMON_SOURCE}
)

add_executable(
    graph-perperty
    ${CMAKE_SOURCE_DIR}/toolkit/property/graph_property.cc
    ${COMMON_SOURCE}
)

add_executable(
    csr-checker
    ${CMAKE_SOURCE_DIR}/toolkit/property/csr_checker.cc
    ${COMMON_SOURCE}
)

add_executable(
    create-alias-table
    ${CMAKE_SOURCE_DIR}/toolkit/weight/create_alias_table.cc
    ${COMMON_SOURCE}
)
add_executable(
    create-prob-prefix-table
    ${CMAKE_SOURCE_DIR}/toolkit/weight/create_prob_prefix_table.cc
    ${COMMON_SOURCE}
)


add_executable(
    train-graph-size
    ${CMAKE_SOURCE_DIR}/toolkit/train_graph_size/train_graph_size.cc
    ${COMMON_SOURCE}
)

add_executable(
    mem-bandwidth
    ${CMAKE_SOURCE_DIR}/toolkit/bandwidth/mem_bandwidth.cc
    ${COMMON_SOURCE}
)

add_executable(
    memcpy-test
    ${CMAKE_SOURCE_DIR}/toolkit/bandwidth/memcpy_test.cc
    ${COMMON_SOURCE}
)
