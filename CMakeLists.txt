cmake_minimum_required(VERSION 3.9)
project(diana-tucker)

list(APPEND CMAKE_PREFIX_PATH "/gpfs/share/software/lapack/3.9.0/gcc/4.8.5/lib/")

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(HOMEBREW_PREFIX "/usr/local/Cellar"
            CACHE PATH "Path to Homebrew installation")
    set(APPLE_SILICON_HOMEBREW_PREFIX "/opt/homebrew"
            CACHE PATH "Path to Homebrew installation")

    list(APPEND CMAKE_PREFIX_PATH "/usr/local")
    list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt")
    list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/lapack")
    list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/openblas")
    list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}/opt/lapack")
    list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}/opt/openblas")
    list(APPEND CMAKE_PREFIX_PATH "${APPLE_SILICON_HOMEBREW_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "${APPLE_SILICON_HOMEBREW_PREFIX}/opt/lapack")
    list(APPEND CMAKE_PREFIX_PATH "${APPLE_SILICON_HOMEBREW_PREFIX}/opt/openblas")
    message(STATUS "${CMAKE_PREFIX_PATH}")

    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)

    list(TRANSFORM CMAKE_PREFIX_PATH APPEND "/include"
            OUTPUT_VARIABLE CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES)
    set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")

    set(CMAKE_FIND_FRAMEWORK NEVER)
    set(CMAKE_FIND_APPBUNDLE NEVER)

    set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)
    set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)
    set(BLA_VENDOR OpenBLAS)
endif (CMAKE_SYSTEM_NAME MATCHES "Darwin")


# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("${cxx17}" COMPILER_SUPPORTS_CXX17)
if (COMPILER_SUPPORTS_CXX17)
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has C++17 support, use C++17.")
    set(CMAKE_CXX_STANDARD 17)
else ()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif (COMPILER_SUPPORTS_CXX17)


# Source files
set(SOURCES
        src/read_data.cpp
        src/util.cpp
        src/operator/operator_float.cpp
        src/operator/operator_double.cpp
        src/communicator.cpp
        src/distribution.cpp
        include/summary.hpp
        src/summary.cpp
        include/algorithm.hpp
        template/algorithm/tucker/hooi_als.tpp
        template/algorithm/sp_tucker/sp_hooi_als.tpp
        template/function/matrix.tpp
        template/function/tensor.tpp
        template/function/sp_tensor.tpp)

# Include directories
include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/template
)


# Set CXX_FLAGS
if (DEBUG_MODE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g3 --Wall")
else(DEBUG_MODE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
endif(DEBUG_MODE)



# define USE_MPI
if (USE_MPI)
    find_package(MPI REQUIRED)
    add_definitions("-DDIANA_MPI")
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    message(STATUS "Found MPI, build MPI support")
endif (USE_MPI)


# define USE_OPENMP
if (USE_OPENMP)
    find_package(OpenMP REQUIRED)
    add_definitions("-DDIANA_OPENMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "Found OpenMP, build OpenMP support")
else ()
    message(STATUS "Will not build OpenMP support")
endif (USE_OPENMP)


# define USE_BLAS
if (USE_BLAS OR USE_LAPACK)
    #TODO
    #We need write our FindBLAS.cmake and FindLapack.cmake
    #Refer to https://github.com/pytorch/pytorch/blob/master/cmake/Modules
    set(BLA_VENDOR "Intel10_64lp")
    find_package(BLAS QUIET)
    if (BLAS_FOUND)
        add_definitions("-DDIANA_MKL")
    endif (BLAS_FOUND)
    if (NOT BLAS_FOUND)
        set(BLA_VENDOR "All")
        find_package(BLAS REQUIRED)
    endif (NOT BLAS_FOUND)
    message(STATUS "FIND THE BLAS OF VERSION ${BLA_VENDOR}")
    add_definitions("-DDIANA_BLAS")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
    message(STATUS "Found BLA_VENDOR=${BLA_VENDOR}, LIB=${BLAS_LIBRARIES}, build BLAS support")
else ()
    message(STATUS "Will not build BLAS support")
endif (USE_BLAS OR USE_LAPACK)


# define USE_LAPACK
if (USE_LAPACK)
    find_package(LAPACK)
    add_definitions("-DDIANA_LAPACK")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
    message(STATUS "Found BLA_VENDOR=${BLA_VENDOR}, LIB=${LAPACK_LIBRARIES}, build LAPACK support")
else ()
    message(STATUS "Will not build LAPACK support")
endif (USE_LAPACK)


add_executable(${PROJECT_NAME} spmain.cpp ${SOURCES})
add_library(${PROJECT_NAME}-lib STATIC ${SOURCES})


if (USE_MPI)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${MPI_C_LIBRARIES})
endif (USE_MPI)

if (USE_OPENMP)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif (USE_OPENMP)

if (USE_BLAS OR USE_LAPACK)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${BLAS_LIBRARIES})
endif (USE_BLAS OR USE_LAPACK)

if (USE_LAPACK)
    target_link_libraries(${PROJECT_NAME} PUBLIC "${LAPACK_LIBRARIES}")
endif (USE_LAPACK)

if (USE_LAPACKE)
    target_link_libraries(${PROJECT_NAME} PUBLIC /usr/lib/x86_64-linux-gnu/liblapacke.a)
endif(USE_LAPACKE)