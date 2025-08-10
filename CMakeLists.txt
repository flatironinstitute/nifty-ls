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
option(NIFTY_LS_HETEROBATCH "Enable heterobatch support with finufft" ON)
option(NIFTY_LS_SANITIZERS "Enable sanitizers for debugging" OFF)
option(NIFTY_LS_COMPILE_WARNING_AS_ERROR "Treat compiler warnings as errors" OFF)

if(NIFTY_LS_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# until https://github.com/wjakob/nanobind/pull/1092 is published
add_compile_options(-Wno-deprecated-literal-operator)

if(NIFTY_LS_SANITIZERS)
    message(STATUS "Enabling nifty-ls sanitizers")
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)

    if(NOT DEFINED FINUFFT_ENABLE_SANITIZERS)
        set(FINUFFT_ENABLE_SANITIZERS ON CACHE BOOL "Enable sanitizers in finufft" FORCE)
    endif()
endif()

set(NIFTY_LS_MODULES cpu_helpers chi2_helpers)

if(NIFTY_LS_HETEROBATCH)
    # Download CPM.cmake if not already available
    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
        file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.42.0/CPM.cmake
             ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
    endif()
    include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
    
    # Add finufft
    CPMAddPackage(
        NAME finufft
        GITHUB_REPOSITORY flatironinstitute/finufft
        GIT_TAG v2.4.1
        OPTIONS
            "FINUFFT_USE_OPENMP ON"
            "FINUFFT_USE_CPU ON"
            "FINUFFT_STATIC_LINKING ON"
            "FINUFFT_POSITION_INDEPENDENT_CODE ON"
            # "FINUFFT_ARCH_FLAGS -march=native"
    )

    list(APPEND NIFTY_LS_MODULES finufft_heterobatch_helpers finufft_chi2_heterobatch_helpers)

endif()


foreach(MODULE_NAME ${NIFTY_LS_MODULES})
    nanobind_add_module(${MODULE_NAME} src/nifty_ls/${MODULE_NAME}.cpp NOMINSIZE STABLE_ABI FREE_THREADED)

    target_compile_options(${MODULE_NAME} PRIVATE
        -Wall -Wextra -std=c++17
    )

    if(NIFTY_LS_COMPILE_WARNING_AS_ERROR)
        set_target_properties(${MODULE_NAME} PROPERTIES COMPILE_WARNING_AS_ERROR TRUE)
    endif()

    if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
        set(OPT_VARS -O3 -ffast-math)
        target_compile_options(${MODULE_NAME} PRIVATE ${OPT_VARS})
    endif()

    if (OpenMP_CXX_FOUND)
        target_link_libraries(${MODULE_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(APPLE)
        # On macOS, use flat namespace to avoid duplicate initialization of libomp
        target_link_options(${MODULE_NAME} PRIVATE -Wl,-flat_namespace)
    endif()


    if(NIFTY_LS_HETEROBATCH AND 
       (${MODULE_NAME} STREQUAL "finufft_heterobatch_helpers" OR 
        ${MODULE_NAME} STREQUAL "finufft_chi2_heterobatch_helpers"))
        target_link_libraries(${MODULE_NAME} PRIVATE finufft)
    endif()

    install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION nifty_ls)
endforeach()
