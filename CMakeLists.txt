cmake_minimum_required(VERSION 3.14)
project(numerical)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)


add_executable(main_app main.cpp)
target_include_directories(main_app PRIVATE src)


add_executable(test_app tests.cpp)
target_include_directories(test_app PRIVATE src)

# Подключаем Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

target_link_libraries(test_app PRIVATE gtest gtest_main OpenMP::OpenMP_CXX)