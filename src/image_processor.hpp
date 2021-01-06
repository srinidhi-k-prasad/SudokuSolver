#ifndef IMAGE_PROCESSOR_H_
#define IMAGE_PROCESSOR_H_

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class ImageProcessor
{
private:
    cv::Mat _origImage, _kernel;
    std::string _path;
    
    void BasicPreProcessing(int gaussianKernel,cv::Mat &kernel,cv::Mat& img);
    int closestNumberDivByNine(int n, int m);
    cv::Rect FindGridContours();
    void ResizeImage(cv::Rect& boundingBox);
       
public:
    cv::Mat procImage;
    std::string save_path = "../results/ProcessedImage.png";
    std::string reprojection_path = "../results/ReprojectedImage.png";
    std::string dataset_path = "../images/digits.png";
    std::string new_dataset_path = "../images/spaced_data.png";
    
    ImageProcessor();
    ImageProcessor(std::string path);
    ~ImageProcessor();   
    
    void ProcessImage();    
    void PrintProperties();
    cv::Mat& getOriginalImage() { return _origImage; }
    
    struct GridProperties{
        cv::Rect boundingBox;
        int cellWidth, cellHeight; 
    } gridProperties;

}; // class

#endif