/* standard external */
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp> 
#include <opencv2/cudaarithm.hpp>
#include <iostream>
#include <set>

/* paralel */
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

/* ccl/buf */
#include <bufapi.h>
#include <bufcore.h>

#define API_COMPACT_BX 256   // threads per block (1D) for compaction kernel
#define MAX_THREADS 1024

/*
 * C U D A - - - section
*/
struct gpubox
{ 
    int x_min, y_min, x_max, y_max; 
};

// kernel: compact label>0 in paralel arrays
__global__ void subframe_compact(const int* labels, size_t step_bytes, int rows, int cols, int* out_labels, int* out_x, int* out_y, unsigned int* out_count)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = (unsigned int)rows * (unsigned int)cols;
    for (unsigned int idx = tid; idx < total; idx += gridDim.x * blockDim.x)
    {
        int r = idx / cols;
        int c = idx % cols;
        const int* row_ptr = (const int*)((const char*)labels + r * step_bytes);
        int lab = row_ptr[c];
        if (lab > 0)
        {
            unsigned int pos = atomicAdd(out_count, 1u);
            out_labels[pos] = lab;
            out_x[pos] = c;
            out_y[pos] = r;
        }
    }
}

// aux function
struct bufmin
{
    __host__ __device__ thrust::tuple<int,int> operator()(const thrust::tuple<int,int>& a,const thrust::tuple<int,int>& b) const
    {
        int av = thrust::get<0>(a);
        int bv = thrust::get<0>(b);
        int au = thrust::get<1>(a);
        int bu = thrust::get<1>(b);
        return thrust::make_tuple(thrust::min(av,bv), thrust::min(au,bu));
    }
};

struct bufmax
{
    __host__ __device__ thrust::tuple<int,int> operator()(const thrust::tuple<int,int>& a, const thrust::tuple<int,int>& b) const
    {
        int av = thrust::get<0>(a);
        int bv = thrust::get<0>(b);
        int au = thrust::get<1>(a);
        int bu = thrust::get<1>(b);
        return thrust::make_tuple(thrust::max(av,bv), thrust::max(au,bu));
    }
};

/*
 * A P I - - - section
 */            
bufapi::bufapi() : ncols(0), nrows(0)
{

}

bufapi::~bufapi()
{

}

void bufapi::img(const cv::cuda::GpuMat& input) 
{
    image = input;
    ncols = image.cols;
    nrows = image.rows;
    labels.create(nrows, ncols, CV_32SC1);
}

cv::cuda::GpuMat bufapi::getlabels() 
{
    if (!image.empty()) 
    {
        int bx = BLOCK_COLS;
        int by = BLOCK_ROWS;
        dim3 block(bx, by);
        int half_cols = (ncols + 1) / 2;
        int half_rows = (nrows + 1) / 2;
        dim3 grid( (half_cols + bx - 1) / bx, (half_rows + by - 1) / by );
        bufcore::InitLabeling<<<grid, block>>>(labels);
        bufcore::Merge<<<grid, block>>>(image, labels);
        bufcore::Compression<<<grid, block>>>(labels);
        bufcore::FinalLabeling<<<grid, block>>>(image, labels);
        cudaDeviceSynchronize();
    }
    return labels;
}
std::vector<cv::Rect> bufapi::getboxes()
{
    std::vector<cv::Rect> result;
    if (labels.empty()) return result;

    double minVal, maxVal;
    cv::cuda::minMax(labels, &minVal, &maxVal);
    int nlabels = static_cast<int>(maxVal);
    if (nlabels == 0) return result;

    int rows = labels.rows;
    int cols = labels.cols;
    unsigned int total_pixels = (unsigned int)rows * (unsigned int)cols;

    // 1) allocate temporary buffers on device (size = number of pixels)
    int *d_lab_compact = nullptr;
    int *d_x_compact = nullptr;
    int *d_y_compact = nullptr;
    unsigned int *d_count = nullptr;

    cudaMalloc((void**)&d_lab_compact, total_pixels * sizeof(int));
    cudaMalloc((void**)&d_x_compact,    total_pixels * sizeof(int));
    cudaMalloc((void**)&d_y_compact,    total_pixels * sizeof(int));
    cudaMalloc((void**)&d_count,       sizeof(unsigned int));
    cudaMemset(d_count, 0, sizeof(unsigned int));

    // 2) launch compaction kernel (1D grid)
    int threads = API_COMPACT_BX;
    if (threads > MAX_THREADS) threads = MAX_THREADS;
    int blocks = (total_pixels + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    subframe_compact<<<blocks, threads>>>( (int*)labels.data, labels.step, rows, cols,
                                           d_lab_compact, d_x_compact, d_y_compact, d_count );
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "subframe_compact error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_lab_compact); cudaFree(d_x_compact); cudaFree(d_y_compact); cudaFree(d_count);
        return result;
    }

    // 3) read compacted amount
    unsigned int host_count = 0;
    cudaMemcpy(&host_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (host_count == 0) {
        cudaFree(d_lab_compact); cudaFree(d_x_compact); cudaFree(d_y_compact); cudaFree(d_count);
        return result;
    }

    // 4) use Thrust on subarrays [0..host_count)
    thrust::device_ptr<int> keys(d_lab_compact);
    thrust::device_ptr<int> vals_x(d_x_compact);
    thrust::device_ptr<int> vals_y(d_y_compact);

    // ordenar por label (key) con zip(x,y) como valores
    thrust::sort_by_key(thrust::device, keys, keys + host_count,
                        thrust::make_zip_iterator(thrust::make_tuple(vals_x, vals_y)));

    // device arrays where Thrust will write unique keys and results
    thrust::device_vector<int> unique_keys(host_count);
    thrust::device_vector<int> out_min_x(host_count);
    thrust::device_vector<int> out_min_y(host_count);
    thrust::device_vector<int> out_max_x(host_count);
    thrust::device_vector<int> out_max_y(host_count);

    // 4a) reduce_by_key for MIN X
    auto ret_minx = thrust::reduce_by_key(thrust::device, keys, keys + host_count, vals_x, unique_keys.begin(), out_min_x.begin(), thrust::equal_to<int>(), thrust::minimum<int>());
    int unique_count_minx = (int)(ret_minx.first - unique_keys.begin());

    // 4b) reduce_by_key for MIN Y (re usage of unique_keys vector)
    auto ret_miny = thrust::reduce_by_key(thrust::device, keys, keys + host_count, vals_y, unique_keys.begin(), out_min_y.begin(), thrust::equal_to<int>(), thrust::minimum<int>());
    int unique_count_miny = (int)(ret_miny.first - unique_keys.begin());

    // 4c) reduce_by_key for MAX X
    auto ret_maxx = thrust::reduce_by_key(thrust::device, keys, keys + host_count, vals_x, unique_keys.begin(), out_max_x.begin(), thrust::equal_to<int>(), thrust::maximum<int>());
    int unique_count_maxx = (int)(ret_maxx.first - unique_keys.begin());

    // 4d) reduce_by_key for MAX Y
    auto ret_maxy = thrust::reduce_by_key(thrust::device, keys, keys + host_count, vals_y, unique_keys.begin(), out_max_y.begin(), thrust::equal_to<int>(), thrust::maximum<int>());
    int unique_count_maxy = (int)(ret_maxy.first - unique_keys.begin());

    // check matching 
    int unique_count = unique_count_minx;
    if (unique_count != unique_count_miny || unique_count != unique_count_maxx || unique_count != unique_count_maxy) 
    {
        unique_count = std::min(std::min(unique_count_minx, unique_count_miny), std::min(unique_count_maxx, unique_count_maxy));
    }

    // 5) output to host
    std::vector<int> h_keys(unique_count);
    std::vector<int> h_minx(unique_count), h_miny(unique_count), h_maxx(unique_count), h_maxy(unique_count);

    thrust::copy(unique_keys.begin(), unique_keys.begin() + unique_count, h_keys.begin());
    thrust::copy(out_min_x.begin(), out_min_x.begin() + unique_count, h_minx.begin());
    thrust::copy(out_min_y.begin(), out_min_y.begin() + unique_count, h_miny.begin());
    thrust::copy(out_max_x.begin(), out_max_x.begin() + unique_count, h_maxx.begin());
    thrust::copy(out_max_y.begin(), out_max_y.begin() + unique_count, h_maxy.begin());

    // 6) rects (label -> rect)
    for (int i = 0; i < unique_count; ++i)
    {
        int lab = h_keys[i];
        if (lab <= 0 || lab > nlabels) continue;
        int xmin = h_minx[i];
        int ymin = h_miny[i];
        int xmax = h_maxx[i];
        int ymax = h_maxy[i];
        if (xmin <= xmax && ymin <= ymax)
            result.emplace_back(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
    }

    // 7) mem dealloc
    cudaFree(d_lab_compact);
    cudaFree(d_x_compact);
    cudaFree(d_y_compact);
    cudaFree(d_count);

    return result;
}

void bufapi::reset() 
{
    image.release();
    labels.release();
    ncols = nrows = 0;
}
