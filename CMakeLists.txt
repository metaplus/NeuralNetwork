cmake_minimum_required(VERSION 3.7)
project(nnet_final)

set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILES
        #main_kmeans.cpp
        liblinear/blas/blas.h   liblinear/blas/dnrm2.c  liblinear/blas/daxpy.c
        liblinear/blas/ddot.c   liblinear/blas/dscal.c  liblinear/tron.h
        liblinear/tron.cpp      liblinear/linear.h      liblinear/linear.cpp
        common/cite.hpp         common/citeSTL.hpp      common/citeBoost.hpp
        literal.hpp             preliminary.hpp  )
add_definitions("-O3")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
find_package(Boost REQUIRED COMPONENTS system filesystem thread)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    add_executable(nnet_linear main_linear.cpp ${SOURCE_FILES})
    add_executable(nnet_minmax main_minmax.cpp ${SOURCE_FILES})
    add_executable(nnet_kmeans main_kmeans.cpp ${SOURCE_FILES})
    set(LIBS ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(nnet_linear ${LIBS})
    target_link_libraries(nnet_minmax ${LIBS})
    target_link_libraries(nnet_kmeans ${LIBS})
endif()
