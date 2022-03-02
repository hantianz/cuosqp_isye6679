cmake_minimum_required(VERSION 3.16)
project(cuosqp_gt CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuosqp_gt main.cu)

set_target_properties(
        cuosqp_gt
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)