OPENCV_DIR = `pkg-config --cflags opencv` -I/usr/include/opencv2
OPENCV_LIBS = `pkg-config --libs opencv`

gpu_match : gpu_match.cpp
	g++ -Wall -ggdb $(OPENCV_DIR) $(OPENCV_LIBS) -o gpu_match gpu_match.cpp
