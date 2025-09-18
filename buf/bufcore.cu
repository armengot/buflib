// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <opencv2/cudafeatures2d.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #include <labeling_algorithms.h>
// #include <register.h>

#define BLOCK_ROWS 16
#define BLOCK_COLS 16

using namespace cv;

namespace {

// Only use it with unsigned numeric types
template <typename T>
__device__ __forceinline__ unsigned char HasBit(T bitmap, unsigned char pos) {
    return (bitmap >> pos) & 1;
}

// Returns the root index of the UFTree
__device__ unsigned Find(const int* s_buf, unsigned n) {
    while (s_buf[n] != n) {
        n = s_buf[n];
    }
    return n;
}

__device__ unsigned FindAndCompress(int* s_buf, unsigned n) {
    unsigned id = n;
    while (s_buf[n] != n) {
        n = s_buf[n];
        s_buf[id] = n;
    }
    return n;
}

// Merges the UFTrees of a and b
__device__ void Union(int* s_buf, unsigned a, unsigned b) {
    bool done;
    do {
        a = Find(s_buf, a);
        b = Find(s_buf, b);
        if (a < b) {
            int old = atomicMin(s_buf + b, a);
            done = (old == b);
            b = old;
        }
        else if (b < a) {
            int old = atomicMin(s_buf + a, b);
            done = (old == a);
            a = old;
        }
        else {
            done = true;
        }
    } while (!done);
}

__global__ void InitLabeling(cuda::PtrStepSzi labels) {
    unsigned row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    unsigned col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    unsigned labels_index = row * (labels.step / labels.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {
        labels[labels_index] = labels_index;
    }
}

__global__ void Merge(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {
    unsigned row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    unsigned col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    unsigned img_index = row * img.step + col;
    unsigned labels_index = row * (labels.step / labels.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {

        unsigned P = 0;
        char buffer alignas(int)[4];
        *(reinterpret_cast<int*>(buffer)) = 0;

        if (col + 1 < img.cols) {
            *(reinterpret_cast<int16_t*>(buffer)) = *(reinterpret_cast<int16_t*>(img.data + img_index));
            if (row + 1 < img.rows) {
                *(reinterpret_cast<int16_t*>(buffer + 2)) = *(reinterpret_cast<int16_t*>(img.data + img_index + img.step));
            }
        }
        else {
            buffer[0] = img.data[img_index];
            if (row + 1 < img.rows) buffer[2] = img.data[img_index + img.step];
        }

        if (buffer[0]) P |= 0x777;
        if (buffer[1]) P |= (0x777 << 1);
        if (buffer[2]) P |= (0x777 << 4);

        if (col == 0) P &= 0xEEEE;
        if (col + 1 >= img.cols) P &= 0x3333;
        else if (col + 2 >= img.cols) P &= 0x7777;

        if (row == 0) P &= 0xFFF0;
        if (row + 1 >= img.rows) P &= 0xFF;

        if (P > 0) {
            if (HasBit(P, 0) && img.data[img_index - img.step - 1]) Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size) - 2);
            if ((HasBit(P, 1) && img.data[img_index - img.step]) || (HasBit(P, 2) && img.data[img_index + 1 - img.step])) Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size));
            if (HasBit(P, 3) && img.data[img_index + 2 - img.step]) Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size) + 2);
            if ((HasBit(P, 4) && img.data[img_index - 1]) || (HasBit(P, 8) && img.data[img_index + img.step - 1])) Union(labels.data, labels_index, labels_index - 2);
        }
    }
}

__global__ void Compression(cuda::PtrStepSzi labels) {
    unsigned row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    unsigned col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    unsigned labels_index = row * (labels.step / labels.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {
        FindAndCompress(labels.data, labels_index);
    }
}

__global__ void FinalLabeling(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {
    unsigned row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    unsigned col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    unsigned labels_index = row * (labels.step / labels.elem_size) + col;
    unsigned img_index = row * (img.step / img.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {

        unsigned int label = labels[labels_index] + 1;

        if (img.data[img_index])
            labels[labels_index] = label;
        else
            labels[labels_index] = 0;

        if (col + 1 < labels.cols) {
            if (img.data[img_index + 1])
                labels[labels_index + 1] = label;
            else
                labels[labels_index + 1] = 0;

            if (row + 1 < labels.rows) {
                if (img.data[img_index + img.step + 1])
                    labels[labels_index + (labels.step / labels.elem_size) + 1] = label;
                else
                    labels[labels_index + (labels.step / labels.elem_size) + 1] = 0;
            }
        }

        if (row + 1 < labels.rows) {
            if (img.data[img_index + img.step])
                labels[labels_index + (labels.step / labels.elem_size)] = label;
            else
                labels[labels_index + (labels.step / labels.elem_size)] = 0;
        }
    }
}

} // namespace

