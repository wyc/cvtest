#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <utility>

using namespace cv;

int main(int argc, char** argv)
{
        
        Mat train_img, query_img;
        Mat result;
        Mat debug_img;

        train_img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
        query_img = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

        gpu::GpuMat train_gpu, query_gpu;
        gpu::GpuMat train_kp_gpu, train_desc_gpu;
        gpu::GpuMat query_kp_gpu, query_desc_gpu;
        gpu::SURF_GPU surf;

        std::vector<KeyPoint> train_kp, query_kp;
        std::vector<float> train_desc, query_desc;
        std::vector<DMatch> matches;
        for (;;) {
                train_gpu.upload(train_img);
                query_gpu.upload(query_img);

                surf(train_gpu, gpu::GpuMat(), train_kp_gpu, train_desc_gpu, false);
                surf(query_gpu, gpu::GpuMat(), query_kp_gpu, query_desc_gpu, false);

                surf.downloadKeypoints(train_kp_gpu, train_kp);
                surf.downloadKeypoints(query_kp_gpu, query_kp);
                surf.downloadDescriptors(train_desc_gpu, train_desc);
                surf.downloadDescriptors(query_desc_gpu, query_desc);

                gpu::BruteForceMatcher_GPU_base matcher;
                matcher.match(train_desc_gpu, query_desc_gpu, matches);

                drawMatches(train_img, train_kp, query_img, query_kp, matches, debug_img);

                imshow("debug_img", debug_img);

                int c = waitKey(0);
                if (c == 'q')
                        break;
        }

        return 0;
}

