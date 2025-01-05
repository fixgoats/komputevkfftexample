cmake_minimum_required(VERSION 3.24)

project(
    komputetest
    VERSION 1.0
    LANGUAGES CXX)

set(VKFFT_BACKEND 0 CACHE STRING "0 - Vulkan, 1 - CUDA, 2 - HIP, 3 - OpenCL, 4 - Level Zero, 5 - Metal")
set(TARGET_NAME test)
set(SOURCES src/main.cpp src/vkhelpers.cpp)
add_executable(${TARGET_NAME} ${SOURCES})
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 20 REQUIRED)
add_definitions(-DVKFFT_BACKEND=${VKFFT_BACKEND})
find_package(Vulkan)
find_package(glslang REQUIRED)
add_subdirectory(kompute)
target_include_directories(${TARGET_NAME} PRIVATE include)
target_precompile_headers(${TARGET_NAME} PRIVATE include/hack.h)

add_definitions(-DVKFFT_BACKEND=${VKFFT_BACKEND})
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

if(supported)
    set_property(TARGET ${TARGET_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
target_compile_definitions(${TARGET_NAME} PUBLIC -DVK_API_VERSION=13)
add_library(VkFFT INTERFACE)
target_include_directories(VkFFT INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/ /usr/include/glslang/Include)
target_link_libraries(VkFFT INTERFACE glslang::MachineIndependent glslang::OSDependent glslang::GenericCodeGen glslang::SPIRV)
target_compile_definitions(VkFFT INTERFACE -DVKFFT_BACKEND=${VKFFT_BACKEND} -DVKFFT_MAX_FFT_DIMENSIONS=3)
target_compile_definitions(${TARGET_NAME} PUBLIC -DVK_API_VERSION=13)
if(MSVC)
  target_compile_options(${TARGET_NAME} PRIVATE /W4 /WX)
else()
  target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()


install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/" DESTINATION include)
install(TARGETS ${TARGET_NAME} DESTINATION bin)
target_link_libraries(${TARGET_NAME}
    kompute::kompute
    Vulkan::Vulkan
    SPIRV
    glslang
    VkFFT
    )

