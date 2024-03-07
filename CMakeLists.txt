cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# TODO: stable ABI?
nanobind_add_module(cpu src/nifty_ls/cpu.cpp NOMINSIZE)

target_compile_options(cpu PRIVATE
    -Wall -Wextra -Werror -std=c++17
)

if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(cpu PRIVATE -Ofast)
endif()

install(TARGETS cpu LIBRARY DESTINATION nifty_ls)
