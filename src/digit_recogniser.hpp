#ifndef DIGIT_RECOGNISER_H_
#define DIGIT_RECOGNISER_H_

/**
 * @basedon: OpenCV Tutorial, Satya Mallick
 * @description: OCR - Handwritten digits
 * @link: https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/
**/

#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include "image_processor.hpp"
#include "train_ocr.hpp"

//Include namespaces
using namespace cv::ml;
using namespace std;
using namespace cv;

class DigitRecogniser
{
private:
    string _trained_model_path;
    Mat _img;
    Ptr<SVM> _SvM;
public:
    DigitRecogniser();
    ~DigitRecogniser();
    
    std::vector<std::vector<int>> PredictDigits(ImageProcessor* imageProcessor, TrainOCR* trainOCR);
    void LoadSubGrids(vector<Mat> &subGrids, ImageProcessor* imageProcessor);
    void LoadDeskewedSubGrids(vector<Mat> &deskewedSubGrids, vector<Mat> &SubGrids, TrainOCR* trainOCR);
    void HOGCompute(vector<vector<float> > &predictHoG, vector<Mat> &deskewedSubGrids, TrainOCR* trainOCR);
    void VectorToMatrix(int descriptor_size,vector<vector<float> > &predictHoG,Mat &predictMat);
    void ReprojectOnImage(string savePath, shared_ptr<vector<vector<int>>> Sudoku, ImageProcessor* imageProcessor);
    Mat& FloodFill(Mat& img, Mat kernel);
    
}; // class

#endif