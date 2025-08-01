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

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED ${SKBUILD_SABI_COMPONENT})
find_package(nanobind CONFIG REQUIRED)

option(NIFTY_LS_OPENMP "Enable OpenMP support in the compiled extensions" ON)

if(NIFTY_LS_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

set(NIFTY_LS_MODULES cpu_helpers chi2_helpers)

# until https://github.com/wjakob/nanobind/pull/1092 is published
add_compile_options(-Wno-deprecated-literal-operator)

foreach(MODULE_NAME ${NIFTY_LS_MODULES})
    nanobind_add_module(${MODULE_NAME} src/nifty_ls/${MODULE_NAME}.cpp NOMINSIZE STABLE_ABI FREE_THREADED)

    target_compile_options(${MODULE_NAME} PRIVATE
        -Wall -Wextra -std=c++17
    )

    if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(OPT_VARS -O3 -ffast-math)
        else()
            set(OPT_VARS -Ofast)
        endif()
        target_compile_options(${MODULE_NAME} PRIVATE ${OPT_VARS})
    endif()

    if (OpenMP_CXX_FOUND)
        target_link_libraries(${MODULE_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(APPLE)
        # On macOS, use flat namespace to avoid duplicate initialization of libomp
        target_link_options(${MODULE_NAME} PRIVATE -Wl,-flat_namespace)
    endif()

    install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION nifty_ls)
endforeach()
