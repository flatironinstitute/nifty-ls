cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

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
option(NIFTY_LS_CUDA "Enable CUDA backend (cuda_helper)" OFF)
set(NIFTY_LS_CUFINUFFT_ROOT "" CACHE PATH "Prefix where cufinufft is installed (include/, lib/)")
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

if(NIFTY_LS_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # cufinufft C API (required for cuda_helper)
    find_library(
        CUFINUFFT_LIBRARY cufinufft
        HINTS ${NIFTY_LS_CUFINUFFT_ROOT} ${NIFTY_LS_CUFINUFFT_ROOT}/lib ${NIFTY_LS_CUFINUFFT_ROOT}/lib64
    )
    find_path(
        CUFINUFFT_INCLUDE_DIR cufinufft.h
        HINTS ${NIFTY_LS_CUFINUFFT_ROOT} ${NIFTY_LS_CUFINUFFT_ROOT}/include
    )
    if(NOT CUFINUFFT_LIBRARY OR NOT CUFINUFFT_INCLUDE_DIR)
        message(FATAL_ERROR "cufinufft library and headers are required when NIFTY_LS_CUDA=ON")
    endif()

    list(APPEND NIFTY_LS_MODULES cuda_helper chi2_cuda_helper)
endif()

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
    set(MODULE_SRC "src/nifty_ls/${MODULE_NAME}.cpp")
    if(${MODULE_NAME} STREQUAL "cuda_helper")
        set(MODULE_SRC "src/nifty_ls/${MODULE_NAME}.cu")
    endif()
    if(${MODULE_NAME} STREQUAL "chi2_cuda_helper")
        set(MODULE_SRC "src/nifty_ls/${MODULE_NAME}.cu")
    endif()

    nanobind_add_module(${MODULE_NAME} ${MODULE_SRC} NOMINSIZE STABLE_ABI FREE_THREADED)

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

    if(${MODULE_NAME} STREQUAL "cuda_helper" OR ${MODULE_NAME} STREQUAL "chi2_cuda_helper")
        # cufinufft.h may live in ${prefix}/include or ${prefix}/include/cufinufft;
        # finufft_common headers sit one level up, so add the parent include dir too.
        get_filename_component(CUFINUFFT_INCLUDE_PARENT ${CUFINUFFT_INCLUDE_DIR} DIRECTORY)
        target_include_directories(
            ${MODULE_NAME}
            PRIVATE
            ${CUFINUFFT_INCLUDE_DIR}
            ${CUFINUFFT_INCLUDE_PARENT}
            ${CUFINUFFT_INCLUDE_PARENT}/finufft_common
            ${CUFINUFFT_INCLUDE_PARENT}/../include
        )
        target_link_libraries(${MODULE_NAME} PRIVATE ${CUFINUFFT_LIBRARY} CUDA::cudart CUDA::cufft CUDA::cublas)
        set_target_properties(
            ${MODULE_NAME}
            PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_ARCHITECTURES native
        )
    endif()

    install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION nifty_ls)
endforeach()
