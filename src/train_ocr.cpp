#include "train_ocr.hpp"

TrainOCR::TrainOCR()
{   
}

TrainOCR::~TrainOCR()
{ 
}

Mat TrainOCR::deskew(Mat& img)
{
    Moments m = moments(img);
    if(abs(m.mu02)< 1e-2) {
        return img.clone();
    }
    
    double skew = m.mu11/m.mu02;
    Mat warpMat = (Mat_<double>(2,3) << 1, skew, -0.5*TrainOCR::getSZ()*skew, 0, 1 , 0);    
    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, imgOut, warpMat, imgOut.size(), WARP_INVERSE_MAP|INTER_LINEAR);
    return img;
} 

void TrainOCR::loadTrainingData(const string &path, vector <Mat> &trainGrids, vector <Mat> &testGrids, vector <int> &trainLabels, vector <int> &testLabels)
{   
    //Load train and test images and labels
    Mat img = imread(path, IMREAD_GRAYSCALE);
    int imgCount = 0;
    for(int i = 0; i < img.rows; i = i + TrainOCR::getSZ())
    {
        for(int j = 0; j < img.cols; j = j + TrainOCR::getSZ())
        {   
            //Read the 20x20 pixel digit image
            Mat digitImg = (img.colRange(j,j+TrainOCR::getSZ()).rowRange(i,i+TrainOCR::getSZ())).clone();
            
            //Add all digit-images in 90% (450-rows) of the columns
            if(j < int(0.9*img.cols)) 
            {
                trainGrids.push_back(digitImg);
            }
            else
            {
                testGrids.push_back(digitImg);
            }
            imgCount++;
        }
    }
    
    cout << "Total number of Images considered: " << imgCount << endl;
    float digitClassID = 0;

    //Classify images into digits with number classes
    for(int z=0;z<int(0.9*imgCount);z++)
    {
        if(z % 450 == 0 && z != 0)
        { 
            digitClassID = digitClassID + 1;
        }
        trainLabels.push_back(digitClassID);
    }
    digitClassID = 0;
    for(int z=0;z<int(0.1*imgCount);z++)
    {
        if(z % 50 == 0 && z != 0)
        {
            digitClassID = digitClassID + 1;
        }
        testLabels.push_back(digitClassID);
    }
}

void TrainOCR::loadDeskewedTrainingData(vector<Mat> &deskewedTrainGrids,vector<Mat> &deskewedTestGrids, vector<Mat> &trainGrids, vector<Mat> &testGrids)
{
    for(int i=0;i<trainGrids.size();i++)
    {
        Mat deskewedImg = deskew(trainGrids[i]);
        deskewedTrainGrids.push_back(deskewedImg);
    }

    for(int i=0;i<testGrids.size();i++)
    {
        Mat deskewedImg = deskew(testGrids[i]);
        deskewedTestGrids.push_back(deskewedImg);
    }    
}

void TrainOCR::loadHoGTrainingData(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, vector<Mat> &deskewedtrainGrids, vector<Mat> &deskewedtestGrids)
{
    for(int y=0;y<deskewedtrainGrids.size();y++){
        vector<float> descriptors;
        
        TrainOCR::HoG.compute(deskewedtrainGrids[y],descriptors);
        trainHOG.push_back(descriptors);
    }

    for(int y=0;y<deskewedtestGrids.size();y++){

        vector<float> descriptors;
        TrainOCR::HoG.compute(deskewedtestGrids[y],descriptors);
        testHOG.push_back(descriptors); ////Note: Size of this descriptor is 81×1 i.e 9x9 concatenated into a row vector
    }
}

void TrainOCR::VectortoMatrix(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, Mat &trainMat, Mat &testMat)
{
    int descriptor_size = trainHOG[0].size();
    for(int i = 0;i<trainHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
           trainMat.at<float>(i,j) = trainHOG[i][j];
        }
    }
    for(int i = 0;i<testHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
            testMat.at<float>(i,j) = testHOG[i][j];
        }
    }
}

void TrainOCR::SVMParams(SVM *SVM)
{
    cout << "Kernel type     : " << SVM->getKernelType() << endl;
    cout << "Type            : " << SVM->getType() << endl;
    cout << "C               : " << SVM->getC() << endl;
    cout << "Degree          : " << SVM->getDegree() << endl;
    cout << "Nu              : " << SVM->getNu() << endl;
    cout << "Gamma           : " << SVM->getGamma() << endl;
}

Ptr<SVM> SVMInit(float C, float gamma)
{
  Ptr<SVM> SVM = SVM::create();
  SVM->setGamma(gamma);
  SVM->setC(C);
  SVM->setKernel(SVM::RBF);
  SVM->setType(SVM::C_SVC);

  return SVM;
}

void TrainOCR::SVMTrain(Ptr<SVM> SVM, Mat &trainMat, vector<int> &trainLabels)
{
  Ptr<TrainData> TrainingData = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
  SVM->train(TrainingData);
  SVM->save(TrainOCR::getSaveModelPath());
}

void TrainOCR::SVMPredict(Ptr<SVM> SVM, Mat &testResponse, Mat &testMat)
{
  SVM->predict(testMat, testResponse);
}

void TrainOCR::SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels)
{
  for(int i = 0; i < testResponse.rows; i++)
  {
    if(testResponse.at<float>(i,0) == testLabels[i])
      count = count + 1;
  }
  accuracy = (count/testResponse.rows)*100;
}

void TrainOCR::TrainSaveModel() 
{
  
    vector <Mat> trainGrids;
    vector <Mat> testGrids;
    vector <int> trainLabels;
    vector <int> testLabels;

    // No data- No training!
    loadTrainingData(TrainOCR::getDatasetPath(),trainGrids,testGrids,trainLabels,testLabels);

    //de-skew images and load training data!
    vector<Mat> deskewedTrainGrids;
    vector<Mat> deskewedTestGrids;
    loadDeskewedTrainingData(deskewedTrainGrids,deskewedTestGrids,trainGrids,testGrids);
    
    //Compute HoG and obtain descriptor!
    vector<vector<float> > trainHOG;
    vector<vector<float> > testHOG;
    loadHoGTrainingData(trainHOG,testHOG,deskewedTrainGrids,deskewedTestGrids);  
    int descriptor_size = trainHOG[0].size();
    cout << "Descriptor Size : " << descriptor_size << endl;

    //Convert 2-D Vector to a Matrix    
    Mat trainMat(trainHOG.size(),descriptor_size,CV_32FC1);
    Mat testMat(testHOG.size(),descriptor_size,CV_32FC1);
    VectortoMatrix(trainHOG,testHOG,trainMat,testMat);
    
    //Init, Train, Test, Predict and Evaluate. 
    Mat testResponse;
    float C = 10; // 12.5
    float gamma = 0.5;
    
    Ptr<SVM> model = SVMInit(C, gamma);
    SVMTrain(model, trainMat, trainLabels);
    SVMPredict(model, testResponse, testMat);

    float count = 0;
    float accuracy = 0 ;
    SVMParams(model);
    SVMevaluate(testResponse, count, accuracy, testLabels);

    cout << "Accuracy :" << accuracy << endl;
}