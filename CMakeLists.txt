cmake_minimum_required(VERSION 3.10)

# meshoptimizer does this too. just don't want any warnings in my output tbh.
# see: https://stackoverflow.com/questions/58708772/cmake-project-in-visual-studio-gives-flag-override-warnings-command-line-warnin
if (POLICY CMP0092)
    cmake_policy(SET CMP0092 NEW)
endif ()

project(mania-mesh-optimizer VERSION 1.0 LANGUAGES CXX)

add_executable(mania-mesh-optimizer
               source/main.cpp)

add_subdirectory(dependencies/meshoptimizer)
target_link_libraries(mania-mesh-optimizer meshoptimizer)

set_property(TARGET mania-mesh-optimizer PROPERTY CXX_STANDARD 20)
target_include_directories(mania-mesh-optimizer PRIVATE dependencies/meshoptimizer/src)
