OPENCV_DIR = `pkg-config --cflags opencv` -I/usr/include/opencv2
OPENCV_LIBS = `pkg-config --libs opencv`

all : imagify


imagify : imagify.cpp
	g++ -Wall -ggdb $(OPENCV_DIR) $(OPENCV_LIBS) -o imagify imagify.cpp
