cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# put a symlink to the compile_commands.json in the build dir
# so the LSP can find it in a known directory without the platform tag
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
        ${CMAKE_CURRENT_BINARY_DIR}/../compile_commands.json
)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED ${SKBUILD_SABI_COMPONENT})
find_package(nanobind CONFIG REQUIRED)

option(NIFTY_LS_OPENMP "Enable OpenMP support in the compiled extensions" ON)

if(NIFTY_LS_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

set(NIFTY_LS_MODULES cpu_helpers nterms_helpers)

foreach(MODULE_NAME ${NIFTY_LS_MODULES})
    nanobind_add_module(${MODULE_NAME} src/nifty_ls/${MODULE_NAME}.cpp NOMINSIZE STABLE_ABI)

    target_compile_options(${MODULE_NAME} PRIVATE
        -Wall -Wextra -std=c++17
    )

    if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
        target_compile_options(${MODULE_NAME} PRIVATE -Ofast)
    endif()

    if (OpenMP_CXX_FOUND)
        target_link_libraries(${MODULE_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION nifty_ls)
endforeach()
