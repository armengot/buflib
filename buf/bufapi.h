#ifndef BUFAPI_H
#define BUFAPI_H

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>


class buf
{
    private:
        cv::cuda::GpuMat image;        
        cv::cuda::GpuMat labels;
        int ncols = 0;
        int nrows = 0;

    public:
        buf();
        ~buf();

        void img(const cv::cuda::GpuMat& input);
        cv::cuda::GpuMat labels();
        void reset();
};

#endif
