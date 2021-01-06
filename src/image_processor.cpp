#include <iostream>
#include "image_processor.hpp"

#define GAUSSIAN_KERNEL 11          // 11
#define ADAP_THRESH_CONSTANT 2      // 2
#define ADAP_THRESH_BLOCKSIZE 5     // 5

ImageProcessor::ImageProcessor()
{
}

ImageProcessor::ImageProcessor(std::string path) : _path(path)
{
    _origImage = cv::imread(_path, cv::IMREAD_GRAYSCALE); // 0:load as grayscale
    procImage = cv::Mat(_origImage.size(), CV_8UC1);
    _kernel = (cv::Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
}

ImageProcessor::~ImageProcessor()
{
}

void ImageProcessor::ProcessImage()
{
    this->BasicPreProcessing(GAUSSIAN_KERNEL, this->_kernel, _origImage);
    cv::Rect boundingBox = this->FindGridContours();
    this->ResizeImage(boundingBox);
}

void ImageProcessor::PrintProperties(void)
{
    std::cout << "Original Image: " << _origImage.size() << "\nCropped Image: " << procImage.size() << std::endl; 
    std::cout << "Grid Properties: \n" 
    << "Top Left: x: " << gridProperties.boundingBox.x << " ,y: " << gridProperties.boundingBox.y << "\n"
    << "          w: " << gridProperties.boundingBox.width << " ,h: " << gridProperties.boundingBox.height << "\n"
    << "         cw: " << gridProperties.cellWidth << " ,ch: " << gridProperties.cellHeight << std::endl;
}

void ImageProcessor::BasicPreProcessing(int gaussianKernel,cv::Mat &kernel,cv::Mat& img)
{
    cv::GaussianBlur(img, procImage, cv::Size(gaussianKernel, 
        gaussianKernel), CV_HAL_BORDER_CONSTANT);

    cv::adaptiveThreshold(procImage, procImage, 255, cv::ADAPTIVE_THRESH_MEAN_C,
        cv::THRESH_BINARY, ADAP_THRESH_BLOCKSIZE, ADAP_THRESH_CONSTANT);
      
    // IMP: If using THRESH_BINARY_INV in adaptive thresh, inversion not needed
    cv::bitwise_not(procImage, procImage);
}

int ImageProcessor::closestNumberDivByNine(int dividend, int divisor = 9) 
{ 
    int quo = dividend / divisor; 
      
    // 1st closest number 
    int n1 = divisor * quo; 
      
    // 2nd closest number 
    int n2 = (divisor * divisor) > 0 ? (dividend * (dividend + 1)) : (divisor * (dividend - 1)); 
      
    // if true, then n1 is closest 
    if (abs(dividend - n1) < abs(dividend - n2)) 
        return n1; 
      
    // else n2 is closest
    return n2;     
}


cv::Rect ImageProcessor::FindGridContours()
{
    int largestArea=0;
    int largestContourIndex=0;
    cv::Rect boundingBox;

    std::vector<std::vector<cv::Point>> contours; // Vector for storing contours

    findContours(procImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE ); // Find the contours in the image
   
    for(size_t i = 0; i < contours.size(); i++ ) // iterate through each contour.
    {
        double area = contourArea(contours[i]);  //  Find the area of contour

        if( area > largestArea )
        {
            largestArea = area;
            largestContourIndex = i;               //Store the index of largest contour
            boundingBox = boundingRect( contours[i] ); // Find the bounding rectangle for biggest contour
        }
    }

    return boundingBox;
}

void ImageProcessor::ResizeImage(cv::Rect& boundingBox){

    // Using bounding box info, crop original image to new dimension
    cv::Mat croppedImg = _origImage(boundingBox);

    // Calculate resize factor, a number close to the cropped image dimension divisible by 9
    int resizeFactor = closestNumberDivByNine(croppedImg.rows, 9);

    // Clear procImage matrix
    procImage = cv::Mat();
       
    cv::resize(croppedImg,procImage,cv::Size(resizeFactor,resizeFactor));

    // Set the grid properties
    gridProperties.boundingBox.x = 0;
    gridProperties.boundingBox.y = 0;
    gridProperties.boundingBox.width = procImage.size().width;
    gridProperties.boundingBox.height = procImage.size().height;
    gridProperties.cellWidth =  gridProperties.boundingBox.width/9;
    gridProperties.cellHeight = gridProperties.boundingBox.height/9;

    this->BasicPreProcessing(11, _kernel, procImage);
    cv::imwrite(save_path, procImage);
}

