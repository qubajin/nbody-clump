cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

project(NBodyClump LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

file(GLOB NBODY_CLUMP_CUDA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_executable(nbody_clump
    src/main.cpp
    ${NBODY_CLUMP_CUDA_SOURCES}
)

target_include_directories(nbody_clump PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
)

target_link_libraries(nbody_clump PRIVATE
    CUDA::cudart
    Boost::filesystem
    Boost::system
)

target_compile_features(nbody_clump PRIVATE cxx_std_14 cuda_std_14)

set_target_properties(nbody_clump PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET nbody_clump PROPERTY
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endif()

if(MSVC)
    target_compile_options(nbody_clump PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
else()
    target_compile_options(nbody_clump PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wno-switch>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-switch>
    )
endif()

add_custom_command(TARGET nbody_clump POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/world.par"
            "$<TARGET_FILE_DIR:nbody_clump>/world.par"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ph.00000000.bt"
            "$<TARGET_FILE_DIR:nbody_clump>/ph.00000000.bt"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ch.00000000.bt"
            "$<TARGET_FILE_DIR:nbody_clump>/ch.00000000.bt"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/src/Rocks"
            "$<TARGET_FILE_DIR:nbody_clump>/Rocks"
    COMMENT "Copying simulation input files"
)
