cmake_minimum_required(VERSION 3.4)

project(greedy_assignment VERSION 1.0
                        DESCRIPTION "A greedy approach to solve the assignment problem"
                        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(pybind11 REQUIRED)

pybind11_add_module(greedy_assignment calc_greedy_assignment.cpp)