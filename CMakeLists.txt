cmake_minimum_required(VERSION 3.27)
project(rvs_laboratories_13 LANGUAGES CXX)

option(BUILD_DATASET_GENERATOR "Build dataset generator" ON)
option(BUILD_MAIN_PROGRAM "Build main program" ON)

if (BUILD_DATASET_GENERATOR)
	set(LIBGPUTK_PATH "/usr/local/libgputk" CACHE STRING "Path to libgputk")

	add_executable(dataset_generator dataset_generator/dataset_generator.cpp)

	target_include_directories(dataset_generator PRIVATE ${LIBGPUTK_PATH}/include)
	target_link_directories(dataset_generator PRIVATE ${LIBGPUTK_PATH}/lib/)
	target_link_libraries(dataset_generator gputk)
endif()

if (BUILD_MAIN_PROGRAM)
	enable_language(CUDA)
	add_executable(rvs_laboratories_13 rvs_laboratories_13/rvs_laboratories_13.cu)
endif()