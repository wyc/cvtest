#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>

#include <cassert>

using namespace cv;

int main(int argc, char** argv) {
        assert(argc >= 2);
        const char* dtype = "SIFT";
        if (argc == 3)
                dtype = "SURF";
                
        Mat img = imread(argv[1]);
        Ptr<FeatureDetector> fdet = FeatureDetector::create(dtype);
        std::vector<KeyPoint> kp;

        fdet->detect(img, kp);

        Ptr<DescriptorExtractor> fext = DescriptorExtractor::create(dtype);

        Mat desc;
        fext->compute(img, kp, desc);

        Mat outimg;
        Scalar kcolor = Scalar(255, 0, 0);
        drawKeypoints(img, kp, outimg, kcolor, DrawMatchesFlags::DEFAULT);

        namedWindow("Output");
        imshow("Output", outimg);

        char c = ' ';
        while ((c = waitKey(0)) != 'q')
                ;

        return 0;
}

