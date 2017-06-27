cmake_minimum_required(VERSION 3.7)
project(nnet_final)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
        main_linear.cpp
        #main_minmax.cpp
        #main_kmeans.cpp
        preliminary.hpp     common/cite.hpp
        common/citeSTL.hpp  common/citeBoost.hpp literal.hpp)
set(EXECUTABLE_OUTPUT_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/bin)
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/liblinea )
find_package(Boost REQUIRED COMPONENTS
        system filesystem thread)

if(Boost_FOUND)
include_directories(${Boost_INCLUDE_DIRS})
add_executable(nnet_final ${SOURCE_FILES})
target_link_libraries(nnet_final
        ${Boost_LIBRARIES}  ${CMAKE_THREAD_LIBS_INIT})
endif()
