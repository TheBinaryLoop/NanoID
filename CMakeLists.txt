cmake_minimum_required(VERSION 3.15)
project(NanoID)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "D:/Source/CPP/libs/opencv/build")
find_package(OpenCV REQUIRED)


FetchContent_Declare(dlib
        GIT_REPOSITORY https://github.com/davisking/dlib.git
        GIT_TAG        v19.19
)
FetchContent_MakeAvailable(dlib)


add_executable(NanoID main.cpp src/System.cpp src/System.h src/Trainer.cpp src/Trainer.h includes/types.h)
target_link_libraries(NanoID dlib::dlib ${OpenCV_LIBS})