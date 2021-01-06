#include "digit_recogniser.hpp"
#include "train_ocr.hpp"
#include "image_processor.hpp"
#include <experimental/filesystem>

DigitRecogniser::DigitRecogniser()
{
    // Define and initialize variables
    _trained_model_path = "../results/SVMClassifierModel.yml";
}

DigitRecogniser::~DigitRecogniser()
{
}

void DigitRecogniser::LoadSubGrids(vector<Mat> &subGrids, ImageProcessor* imageProcessor)
{   
    _img = imread(imageProcessor->save_path, IMREAD_GRAYSCALE);
    // TODO: change to _img = imageProcessor->procImage; 
    bool showDigit = false;
    int imgCount = 0;
    int n = 20; // What is n 

    for(int i = 0; i < _img.rows; i+= imageProcessor->gridProperties.cellHeight)
    {
        for(int j = 0; j < _img.cols; j+=imageProcessor->gridProperties.cellWidth)
        {   
            cv::Mat kernel;
            kernel = (cv::Mat_<uchar>(3,3) << 0,0,0,0,n,0,0,0,0); // TODO

            //Read the digit image
            Mat digitImg = (_img.colRange(j,j+imageProcessor->gridProperties.cellWidth).rowRange(i,i+ imageProcessor->gridProperties.cellHeight)).clone(); // Error line

            cv::Rect myROI( 5, 5, 
                            imageProcessor->gridProperties.cellWidth - 10, 
                            imageProcessor->gridProperties.cellWidth - 10);
            
            // Crop as per ROI
            digitImg = digitImg(myROI);

            if(showDigit){
                    cv::imshow("Digit", digitImg);
                    cv::waitKey(0);
            }
            
            resize(digitImg,digitImg,Size(20,20)); // As training set was 20x20 
            cv::erode(digitImg, digitImg, kernel);
            
            subGrids.push_back(digitImg); // No-inversion required
            
            //Add all digit-images - totals 81 subgrids
            imgCount++;
        }
    }
    cout << "Total number of sub-grids detected: " << imgCount <<  endl;
}

void DigitRecogniser::LoadDeskewedSubGrids(vector<Mat> &deskewedSubGrids, vector<Mat> &SubGrids, TrainOCR* trainOCR)
{   
    
    for(int i=0;i<SubGrids.size();i++)
    {
        Mat deskewedImg = trainOCR->deskew(SubGrids[i]);        
        deskewedSubGrids.push_back(deskewedImg);
    }  
    cout << "Total number of de-skewed images: " << deskewedSubGrids.size() <<  endl;
}

void DigitRecogniser::HOGCompute(vector<vector<float> > &predictHoG, vector<Mat> &deskewedSubGrids, TrainOCR* trainOCR)
{
    for(int y=0;y<deskewedSubGrids.size();y++)
    {
        vector<float> descriptors;
        trainOCR->HoG.compute(deskewedSubGrids[y],descriptors);        
        predictHoG.push_back(descriptors);
    }
    cout << "HoGs Computed!" << endl;
}

void DigitRecogniser::VectorToMatrix(int descriptor_size,vector<vector<float> > &predictHoG,Mat &predictMat)
{
    for(int i = 0;i<predictHoG.size();i++)
    {
        for(int j = 0;j<descriptor_size;j++)
        {
           predictMat.at<float>(i,j) = predictHoG[i][j];
        }
    }
}

void DigitRecogniser::ReprojectOnImage(std::string savePath, shared_ptr<vector<vector<int>>> Sudoku, ImageProcessor* imageProcessor)
{
    cv::Mat img = cv::imread(savePath, 0);
    // int new_count = 0;
    cvtColor(img,img,COLOR_GRAY2RGB);
    int rowCount{0};
    for(int i = 0; i < img.rows; i = i + imageProcessor->gridProperties.cellWidth)
    {int colCount{0};
        for(int j = 0; j < img.cols; j = j + imageProcessor->gridProperties.cellHeight)
        {  
            putText(img, to_string(Sudoku->at(rowCount).at(colCount)), Point(j+5,i+45), FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(0,0,255), 2);
            colCount++;
        }
        rowCount++;
    }
    imwrite(imageProcessor->reprojection_path, img);
}

std::vector<std::vector<int>> DigitRecogniser::PredictDigits(ImageProcessor* imageProcessor, TrainOCR* trainOCR)
{  
    bool modelExists = std::experimental::filesystem::exists(_trained_model_path);
    
    //Check if SVM Trained-model exists. Otherwise, re-train model.
    if(modelExists) {
        _SvM = Algorithm::load<SVM>(_trained_model_path);
        cout << "Found OCR Trained Model! No re-training necessary..\n"; 
        }
        
    else {
        cout << "Could not find an existing trained-model. Program will train the OCR-network before character-recognition..\n";
        trainOCR->TrainSaveModel();
        _SvM = Algorithm::load<SVM>(_trained_model_path);
        cout << "Training Successful!\n"; }
    
    vector <Mat> subGrids;
    LoadSubGrids(subGrids, imageProcessor);

    vector<Mat> deskewedSubGrids;
    LoadDeskewedSubGrids(deskewedSubGrids, subGrids, trainOCR);
       
    vector<vector<float> > predictHoG;
    HOGCompute(predictHoG, deskewedSubGrids, trainOCR);

    int descriptor_size = _SvM->getVarCount();
    Mat predictMat(predictHoG.size(),descriptor_size,CV_32FC1);
    VectorToMatrix(descriptor_size, predictHoG, predictMat);

    Mat testResponse;

    _SvM->predict(predictMat, testResponse);

    vector<vector<int>> sudoku;
    int inc = 0;
    for (int i=0; i < testResponse.rows/9; i++)
    {
        vector<int> row{};
    
        for(int j = 0; j < 9; j++){
            row.push_back(testResponse.at<float>(inc,0));
            inc++;
        }
        sudoku.push_back(row);
    }
    return sudoku;
}

