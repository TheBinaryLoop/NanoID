cmake_minimum_required(VERSION 3.15)
project(NanoID)

set(CMAKE_CXX_STANDARD 17)
set(opencv_DIR "D:/Source/CPP/libs/opencv/build")
find_package(opencv REQUIRED)
#find_package(dlib 19.19.0 CONFIG REQUIRED)
include(FetchContent)
FetchContent_Declare(dlib
        GIT_REPOSITORY https://github.com/davisking/dlib.git
        GIT_TAG        v19.19
)
FetchContent_MakeAvailable(dlib)


add_executable(NanoID main.cpp src/System.cpp src/System.h)
target_link_libraries(NanoID dlib::dlib)