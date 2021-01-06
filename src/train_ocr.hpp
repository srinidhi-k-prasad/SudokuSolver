#ifndef TRAIN_OCR_H_
#define TRAIN_OCR_H_

/**
 * @basedon: OpenCV Tutorial, Satya Mallick
 * @description: OCR - Handwritten digits
 * @link: https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/
**/

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>

//Include namespaces
using namespace cv::ml;
using namespace std;
using namespace cv;

class TrainOCR
{
  private:
    //Define and initialize training parameters
    static const string _dataset_path;
    static const string _save_path;  
    static const int _sz = 20; //No. of pixels in the image.size() of a digit
        
  public:
    TrainOCR();
    ~TrainOCR();

    static const int getSZ(){ return _sz;}

    //Define getters for static strings
    static const string& getDatasetPath()
    {
        static const string _dataset_path = "../images/spaced_data.png";
        return _dataset_path;
    }
    
    static const string& getSaveModelPath()
    {
        static const string _save_path = "../results/SVMClassifierModel.yml";
        return _save_path;
    }
    
    static Mat deskew(Mat& img);
    void loadTrainingData(const string &path, vector <Mat> &trainGrids, vector <Mat> &testGrids, vector <int> &trainLabels, vector <int> &testLabels);
    void loadDeskewedTrainingData(vector<Mat> &deskewedTrainGrids,vector<Mat> &deskewedTestGrids, vector<Mat> &trainGrids, vector<Mat> &testGrids);
    void loadHoGTrainingData(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, vector<Mat> &deskewedtrainGrids, vector<Mat> &deskewedtestGrids);
    void VectortoMatrix(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, Mat &trainMat, Mat &testMat);
    void SVMParams(SVM *SVM);
    void SVMTrain(Ptr<SVM> SVM, Mat &trainMat, vector<int> &trainLabels);
    void SVMPredict(Ptr<SVM> SVM, Mat &testResponse, Mat &testMat);
    void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels);
    void TrainSaveModel(); 
            
    HOGDescriptor HoG
    {
        /**
         * @description: Histogram of Oriented Gradients (Feature descriptor)
         * Steps: Crop to feature --> Calculate gradient magnitude/orientation --> Create Histogram
         * Link: https://www.learnopencv.com/histogram-of-oriented-gradients/
         **/

        Size(20,20), //winSize - Size of the digit images in our dataset is 20×20
        
        Size(8,8), /**blocksize: The notion of blocks exist to tackle illumination variation.
        A large block size makes local changes less significant while a smaller block size weights local changes more.
        Typically blockSize is set to 2 x Gridsize.**/ 
        
        Size(4,4), /**blockStride: The blockStride determines the overlap between neighboring blocks and controls the degree of contrast normalization.
        Typically a blockStride is set to 50% of blockSize.**/

        Size(8,8), /**Gridsize - Chosen based on the scale of the features important to do the classification.
        Size of 10x10 works too.**/
        
        9, /**nbins:  nbins sets the number of bins in the histogram of gradients. The authors of the HOG paper had recommended a value of 9 
        to capture gradients between 0 and 180 degrees in 20 degrees increments.**/

        1,   //derivAper,
        -1,  //winSigma,
        cv::HOGDescriptor::HistogramNormType::L2Hys, //histogramNormType,
        0.2, //L2HysThresh,
        0,   //gammal correction,
        64,  //nlevels=64
        1
    };

}; // Class

#endif