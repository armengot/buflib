/* external standard */
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

/* BUF/CCL api class */
#include <bufapi.h>

int main(int argc, char **argv) 
{
    std::vector<std::string> filenames;
    std::vector<std::string> outnames;

    for (int i = 0; i < 10; ++i)
    {
        std::string filename = "../sample/sample" + std::to_string(i) + ".jpg";
        std::string outname = "output" + std::to_string(i) + ".png";
        filenames.push_back(filename);
        outnames.push_back(outname);
    }
   
    bufapi buf;


    for(unsigned int i=0;i<filenames.size();i++)
    {   
        buf.reset();

        cv::Mat output,image = cv::imread(filenames[i], cv::IMREAD_GRAYSCALE);
        cv::cuda::GpuMat gpuimg;

        if (!image.empty())
            gpuimg.upload(image);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        buf.img(gpuimg);
        buf.getlabels();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()/1000.0;
        std::cout << "[BUF] processing image [" << image.cols << "x" << image.rows << "] with " << filenames[i] << " took " << duration << " milliseconds." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        std::vector<cv::Rect> boxes = buf.getboxes();
        end_time = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()/1000.0;
        std::cout << "[API] getting boxes    [" << image.cols << "x" << image.rows << "] with " << filenames[i] << " took " << duration2 << " milliseconds. TOTAL: " << duration+duration2 << std::endl;
        
        /* TV show 
        cv::cvtColor(image, output, cv::COLOR_GRAY2BGR);
        for (const auto& r : boxes) 
        {
            cv::rectangle(output, r, cv::Scalar(0, 255, 0), 1); // Verde
        }
        cv::Mat small;
        double scale = 0.4f;
        cv::resize(output,small,cv::Size(), scale, scale, cv::INTER_AREA);
        cv::imshow(filenames[i], small);
        cv::waitKey(0);        
        cv::destroyWindow(filenames[i]);  
        */
        
    }
    std::cout << std::endl;

    return 0;
}
