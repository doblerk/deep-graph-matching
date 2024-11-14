cmake_minimum_required(VERSION 3.4)

project(greedy_assignment VERSION 1.0
                        DESCRIPTION "A greedy approach to solve the assignment problem"
                        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../gnn_ged/assignment")

find_package(pybind11 REQUIRED)

pybind11_add_module(greedy_assignment calc_greedy_assignment.cpp)

set_target_properties(greedy_assignment PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../gnn_ged/assignment"
)