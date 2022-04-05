cmake_minimum_required(VERSION 3.16)
project(cuosqp_isye6679 C)

set(CMAKE_C_STANDARD 99)

add_executable(cpu cpu_verison.c)
target_link_libraries(cpu m)

#project(cuosqp_isye6679 CUDA)
#
#set(CMAKE_CUDA_STANDARD 14)
#
#add_executable(cuosqp_gt main.cu cpu_version.cpp)
#
#set_target_properties(
#        cuosqp_gt
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)