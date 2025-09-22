#ifndef BUFCORE_H
#define BUFCORE_H

#include <opencv2/core/cuda.hpp>

namespace bufcore 
{
    #define BLOCK_ROWS 16
    #define BLOCK_COLS 16

    __global__ void InitLabeling(cv::cuda::PtrStepSzi labels);
    __global__ void Merge(const cv::cuda::PtrStepSzb img, cv::cuda::PtrStepSzi labels);
    __global__ void Compression(cv::cuda::PtrStepSzi labels);
    __global__ void FinalLabeling(const cv::cuda::PtrStepSzb img, cv::cuda::PtrStepSzi labels);
}


#endif
