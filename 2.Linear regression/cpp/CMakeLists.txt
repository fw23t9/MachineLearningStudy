cmake_minimum_required(VERSION 3.3)

PROJECT (LinearRegression_cpp)

file(GLOB SOURCE_FILES src/*.*)
add_executable(LinearRegression ${SOURCE_FILES})