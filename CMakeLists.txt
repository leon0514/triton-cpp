# Root CMakeLists.txt for triton-cpp project
#
# Build the Triton custom backend shared libraries (preprocess, yolo11_postprocess, yolov5_postprocess).
#
# Recommended build environment:
#   Docker image: nvcr.io/nvidia/tritonserver:25.01-py3
#
# Build instructions:
#   mkdir build && cd build
#   cmake -DTRITON_DIR=/opt/tritonserver .. -DCMAKE_CUDA_ARCHITECTURES=80
#   make -j$(nproc)
#
# The resulting shared libraries should be deployed as:
#   <model_repository>/<model_name>/1/libtriton_<backend>.so
# or registered in the Triton backend directory:
#   <backend_dir>/<backend>/libtriton_<backend>.so

cmake_minimum_required(VERSION 3.18)
project(triton_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Triton Backend API headers
# Search common installation locations inside SDK images and custom builds.
set(TRITON_DIR "" CACHE PATH "Path to Triton Inference Server installation")

if(NOT TRITON_DIR)
    find_path(TRITON_BACKEND_HEADER_DIR
        NAMES triton/core/tritonbackend.h
        PATHS
            /workspace/install/include
            /workspace/install/include/triton
            /opt/tritonserver/include
            /opt/tritonserver/include/triton
            /usr/include
            /usr/local/include
        NO_DEFAULT_PATH
    )

    if(TRITON_BACKEND_HEADER_DIR)
        # TRITON_BACKEND_HEADER_DIR is the directory containing 'triton/core/tritonbackend.h'
        # We need the parent of 'triton' as the installation prefix.
        get_filename_component(TRITON_DIR ${TRITON_BACKEND_HEADER_DIR} DIRECTORY)
    endif()
endif()

if(TRITON_DIR AND EXISTS ${TRITON_DIR}/include/triton/core/tritonbackend.h)
    message(STATUS "Found Triton installation at: ${TRITON_DIR}")
elseif(TRITON_DIR)
    message(FATAL_ERROR "TRITON_DIR ${TRITON_DIR} does not contain include/triton/core/tritonbackend.h")
else()
    message(FATAL_ERROR "Could not find Triton backend headers (triton/core/tritonbackend.h). "
                        "Please use the Triton runtime image (e.g. nvcr.io/nvidia/tritonserver:25.01-py3) "
                        "or set TRITON_DIR to the Triton installation prefix.")
endif()

# Include directories
set(COMMON_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRITON_DIR}/include
    ${TRITON_DIR}/include/triton
)

include_directories(${COMMON_INCLUDE_DIRS})

set(COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/memory.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/norm.cpp
)

# -------------------- Preprocess backend --------------------
set(PREPROCESS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/preprocess_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/preprocess_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/triton_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/triton_backend.cpp
)

add_library(triton_preprocess SHARED ${PREPROCESS_SOURCES} ${COMMON_SOURCES})
set_target_properties(triton_preprocess PROPERTIES
    OUTPUT_NAME "triton_preprocess"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
if(CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET triton_preprocess PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()
target_compile_options(triton_preprocess PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
target_link_libraries(triton_preprocess CUDA::cudart Threads::Threads)

# -------------------- YOLO11 postprocess backend --------------------
set(YOLO11_POSTPROCESS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo11_postprocess/yolo11_postprocess_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo11_postprocess/yolo11_postprocess_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo11_postprocess/triton_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo11_postprocess/triton_backend.cpp
)

add_library(triton_yolo11_postprocess SHARED ${YOLO11_POSTPROCESS_SOURCES} ${COMMON_SOURCES})
set_target_properties(triton_yolo11_postprocess PROPERTIES
    OUTPUT_NAME "triton_yolo11_postprocess"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
if(CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET triton_yolo11_postprocess PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()
target_compile_options(triton_yolo11_postprocess PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
target_link_libraries(triton_yolo11_postprocess CUDA::cudart Threads::Threads)

# -------------------- YOLOV5 postprocess backend --------------------
set(YOLOV5_POSTPROCESS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolov5_postprocess/yolov5_postprocess_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolov5_postprocess/yolov5_postprocess_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolov5_postprocess/triton_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolov5_postprocess/triton_backend.cpp
)

add_library(triton_yolov5_postprocess SHARED ${YOLOV5_POSTPROCESS_SOURCES} ${COMMON_SOURCES})
set_target_properties(triton_yolov5_postprocess PROPERTIES
    OUTPUT_NAME "triton_yolov5_postprocess"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
if(CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET triton_yolov5_postprocess PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()
target_compile_options(triton_yolov5_postprocess PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
target_link_libraries(triton_yolov5_postprocess CUDA::cudart Threads::Threads)

# Triton backend libraries
if(TRITON_DIR)
    find_library(TRITON_BACKEND_API_LIB
        NAMES tritonbackend
        PATHS ${TRITON_DIR}/lib ${TRITON_DIR}/lib64
        NO_DEFAULT_PATH
    )
    find_library(TRITON_SERVER_API_LIB
        NAMES tritonserver
        PATHS ${TRITON_DIR}/lib ${TRITON_DIR}/lib64
        NO_DEFAULT_PATH
    )

    if(TRITON_BACKEND_API_LIB)
        target_link_libraries(triton_preprocess ${TRITON_BACKEND_API_LIB})
        target_link_libraries(triton_yolo11_postprocess ${TRITON_BACKEND_API_LIB})
        target_link_libraries(triton_yolov5_postprocess ${TRITON_BACKEND_API_LIB})
    endif()
    if(TRITON_SERVER_API_LIB)
        target_link_libraries(triton_preprocess ${TRITON_SERVER_API_LIB})
        target_link_libraries(triton_yolo11_postprocess ${TRITON_SERVER_API_LIB})
        target_link_libraries(triton_yolov5_postprocess ${TRITON_SERVER_API_LIB})
    endif()
endif()

install(TARGETS triton_preprocess triton_yolo11_postprocess triton_yolov5_postprocess
    LIBRARY DESTINATION lib
)
