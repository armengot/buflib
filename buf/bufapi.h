#ifndef BUFAPI_H
#define BUFAPI_H

/* external standard */
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

class bufapi
{
    private:
        cv::cuda::GpuMat image;        
        cv::cuda::GpuMat labels;
        int ncols = 0;
        int nrows = 0;

    public:
        bufapi();
        ~bufapi();

        void img(const cv::cuda::GpuMat& input);
        cv::cuda::GpuMat getlabels();
        std::vector<cv::Rect> getboxes();
        void reset();
};

#endif
