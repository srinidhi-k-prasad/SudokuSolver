## **Sudoku Solver**

![Semantic description of image](/source/images/results.png "Sudoku-Solved")

### **Description:**

A simple C++ project, that aims to: 

- Read and pre-process an image for [Optical Character Recognition](https://github.com/srinidhi-k-prasad/OCR.git)
- Solve a standard Sudoku puzzle 

### **Contents:**

The contents of the program can be best described as follows:

* `ImageProcessor` handles the pre-processing steps needed to identify the Sudoku grid in an image, localise it and extract the Region of Interest for digit recognition. 
* `DigitRecogniser` accepts the processed image with the Sudoku grid to classify the digits and store them in a `vector<vector<int>>`.
* `TrainOCR`, runs for the first time if the trained model is not present. It used Histogram of Oriented Gradients (HoG) and Support Vector Machines (SVM) to train the classifier on the MNIST dataset. 
* `Sudoku` , it contains the logic for solving the puzzle based on backtracking algorithm. It accepts the vector of identified digits and after solving returns a vector of int's as the solution which is printed on the console output.

### Program Flow

`main.cpp`, as the name suggests, is the main-entry point for the application.

* The program first checks if the image was specified as the second argument during execution. If not, it raises a helpful error to the user.
* The image is then passed to the `ImageProcessor` object to perform pre-processing operations such as blur or edge-detection to identifiy the outer grid of the sudoku-puzzle.
* Then, the objects for the `TrainOCR` & `DigitRecogniser` are created to which, the pre-processed image is passed as an argument. 
* Post digit-classification, a vector is returned by `DigitRecogniser` class which is then passed on to the `Sudoku` object calling the `SolveBoard` function for a solution.
* If a solution exists, it is re-projected on a duplicate of the image and displayed as a console output. If not solution exists, the string "Solution not found" is returned as the output.

### **Dependencies**

* cmake >= 3.11.3
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
* OpenCV >= 4.1
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
	
### **Install instructions:**

1. Make sure to meet dependencies.

2. Git clone the project and create a folder `build` in the top-level directory.

```console
user@pc:~$ git clone https://github.com/srinidhi-k-prasad/SudokuSolver.git 
```

3. build your project
	
```console
user@pc:~$ SudokuSolver/build/cmake..
user@pc:~$ SudokuSolver/build/make
```

4. Run project
```console
user@pc:~$ SudokuSolver/build/./sudoko <path_to_the_sudoku_image>
```

### **Known Issues:**

It is of importance to declare that the implementation has only been addressed for the input test-image. 

1. So, the classifier is not robust. It may wrongly classify the digits. 
	
2. Blank spaces are sometimes misclassified, if they that contain that stray pixel.
	
### **Acknowledgements**

Based on the open-source tutorial from Sathya Mallick:

- Handwritten Digits Classification : An OpenCV ( C++ / Python ) Tutorial ([Visit](https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/))

## Future Extensions

1. The implementation can be extented to a variety of sudoku-images.
2. Implement real-time solution to a puzzle using a camera/video stream.