cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(OpenMP)

# TODO: stable ABI?
nanobind_add_module(cpu_helpers src/nifty_ls/cpu_helpers.cpp NOMINSIZE)

target_compile_options(cpu_helpers PRIVATE
    -Wall -Wextra -Werror -std=c++17
)

if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(cpu_helpers PRIVATE -Ofast)
endif()

if (OpenMP_CXX_FOUND)
    target_link_libraries(cpu_helpers PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS cpu_helpers LIBRARY DESTINATION nifty_ls)
