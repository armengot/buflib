#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>

#include <bufcore.cu>

buf::buf() 
{
    ncols = 0;
    nrows = 0;
}

buf::~buf() 
{
}

void buf::img(const cv::cuda::GpuMat& input) 
{
    image = input;
    ncols = image.cols;
    nrows = image.rows;
    labels.create(nrows, ncols, CV_32SC1);  // preparamos buffer de etiquetas
}

cv::cuda::GpuMat buf::labels() 
{
    if (!image.empty()) 
    {
        BUF bufCore;
        bufCore.setImage(image);
        bufCore.PerformLabeling();
        labels = bufCore.getLabels();
    }
    return labels;
}

void buf::reset() 
{
    image.release();
    labels.release();
    ncols = 0;
    nrows = 0;
}
