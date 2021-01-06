#include <iostream>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "image_processor.hpp"
#include "sudoku.hpp"
#include "digit_recogniser.hpp"
#include "train_ocr.hpp"


int main(int argc, char* argv[])
{
    if (argc < 2){
        std::cerr << "\x1B[31mERROR: \033[0m Incorrect arguments given!\n";
        std::cerr << "Usage: " << argv[0] << " IMAGE_PATH" << "\n";
        return 1;
    }

    // Preprocess image to extract outer grid and pass this cropped image to 
    ImageProcessor image(argv[1]);
    image.ProcessImage();
    image.PrintProperties();

    // Pass processed image to the classifier object which returns a vector<vector<int>> 
    DigitRecogniser digits;
    TrainOCR trainOCR;
    std::shared_ptr<std::vector<std::vector<int>>> unsolved;
    unsolved = std::make_shared<std::vector<std::vector<int>>>(digits.PredictDigits(&image, &trainOCR));

    // Image with classifications
    digits.ReprojectOnImage(image.save_path, unsolved, &image);
    
    // Solve Sudoku
    Sudoku board(unsolved);
    if(board.SolveBoard() == true){
        std::shared_ptr<std::vector<std::vector<int>>> solved = board.getSolution();
        board.PrintBoard();
        // Reproject the solution
        digits.ReprojectOnImage(image.save_path, solved, &image);
    }
    else
    {
        std::cout << "No solution exists!" << std::endl;
        return -1;
    }

    return 0;
}

