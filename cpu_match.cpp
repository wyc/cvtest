#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;

int main(int argc, char** argv) {
        if (argc != 3)
                return -1;

        Mat raw_im1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
        Mat raw_im2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

        if (!raw_im1.data || !raw_im2.data)
                return -1;

        Mat im1, im2;
        pyrDown(raw_im1, im1, Size(raw_im1.cols / 2, raw_im1.rows / 2));
        pyrDown(raw_im2, im2, Size(raw_im2.cols / 2, raw_im2.rows / 2));
        imshow("im1", im1);
        waitKey(0);
        imshow("im2", im2);

        waitKey(0);

        int minHessian = 400;
        SurfFeatureDetector det(minHessian);

        std::vector<KeyPoint> kp1, kp2;

        det.detect(im1, kp1);
        det.detect(im2, kp2);

        SurfDescriptorExtractor ext;

        Mat desc1, desc2;

        ext.compute(im1, kp1, desc1);
        ext.compute(im2, kp2, desc2);

        BruteForceMatcher<L2<float> > matcher;
        std::vector<DMatch> matches;
        matcher.match(desc1, desc2, matches);

        std::sort(matches.begin(), matches.end());
        //matches.resize(30);

        Mat im_matches;
        drawMatches(im1, kp1, im2, kp2, matches, im_matches);

        imshow("Matches", im_matches);

        waitKey(0);


        return 0;
}

