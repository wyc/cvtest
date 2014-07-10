#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <utility>

using namespace cv;
VideoCapture* vc = NULL;

int main(int argc, char** argv)
{
        int frames, width, height, fps, frame_num;

        vc = new VideoCapture(0);
        height = (int) vc->get(CV_CAP_PROP_FRAME_HEIGHT);
        width = (int) vc->get(CV_CAP_PROP_FRAME_WIDTH);

        std::cout << "width:\t\t" << width << std::endl
                  << "height:\t\t" << height << std::endl;

        namedWindow("video", 1);
        Mat frame, img;
        for (frame_num = 0; ; frame_num++) {
                //std::cout << "Processing Frame " << frame_num << std::endl;

                *vc >> frame;

                if (frame.empty())
                        break;

                cvtColor(frame, img,CV_BGR2GRAY);
                

                Point max(-1, -1);
                double max_pixel = -1;
                for (int x = 1; x < img.cols - 1; x++) {
                        for (int y = 1; y < img.rows - 1; y++) {
                                Mat sub = img.rowRange(y - 1, y + 1).colRange(x - 1, x + 1);
                                //int pixel = static_cast<int>(img.at<uchar>(y,x));
                                double pixel = sum(sub)[0];
                                if (pixel > max_pixel) {
                                        max_pixel = pixel;
                                        max.x = x;
                                        max.y = y;
                                }
                        }
                }
                std::cout << "Offset: " << max - Point(width/2, height/2) << std::endl;

                circle(frame, max, 3, CV_RGB(0, 255, 0), 2);
                imshow("video", frame);
                waitKey(20);
        }

        delete vc;
        return 0;
}

