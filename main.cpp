#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include <bufapi.h>

int main() 
{

    cv::Mat img = cv::imread("test.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) 
    {
        std::cerr << "No se pudo abrir la imagen\n";
        return -1;
    }   
    cv::cuda::GpuMat d_img;
    d_img.upload(img);    
    buf b;
    b.img(d_img);    
    cv::cuda::GpuMat d_labels = b.labels();    
    cv::Mat labels;
    d_labels.download(labels);    
    cv::imshow("Labels", labels * 10);
    cv::waitKey(0);
    return 0;
}
