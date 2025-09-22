/* external standard */
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

/* BUF/CCL api class */
#include <bufapi.h>

int main(int argc, char** argv) 
{
    if (argc<2)
    {
        std::cerr << "Usage: " << argv[0] << " <file path>" << std::endl;
        return -1;
    }
    std::string filename = argv[1];
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (img.empty()) 
    {
        std::cerr << "Couldnt open the image.\n";
        return -1;
    }   
    cv::imshow("SOURCE",img);
    cv::waitKey(0);
    cv::cuda::GpuMat d_img;
    d_img.upload(img);    
    bufapi b;
    b.img(d_img);    
    cv::cuda::GpuMat d_labels = b.getlabels();    
    cv::Mat labels;
    d_labels.download(labels);    
    cv::Mat labels_8u;
    labels.convertTo(labels_8u, CV_8U, 10);
    std::vector<cv::Rect> boxes = b.getboxes();
    cv::Mat labels_bgr;
    cv::cvtColor(labels_8u, labels_bgr, cv::COLOR_GRAY2BGR);
    int i = 0;
    for (const auto& r : boxes) 
    {
        cv::rectangle(labels_bgr, r, cv::Scalar(0, 255, 0), 1);
        std::cout << "BOX(" << i << "): [" << r.x << "," << r.y << "] W = " << r.width << " H = " << r.height << std::endl;
        i++;
    }
    cv::imshow("Labels", labels_bgr);
    cv::waitKey(0);
    cv::imwrite("marv_test.png",labels_bgr);
    return 0;
}
