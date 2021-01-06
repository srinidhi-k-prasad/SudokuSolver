#include "sudoku.hpp"

Sudoku::Sudoku(std::shared_ptr<std::vector<std::vector<int>>> inputGrid){
    _sudokuOutput = inputGrid;
}

Sudoku::~Sudoku()
{
}

bool Sudoku::RowCheck(int row, int num){
    for(int col = 0; col < _gridSize ; col++){
        if(_sudokuOutput->at(row).at(col) == num)
            return true;
    }
    return false;
}

bool Sudoku::ColumnCheck(int col, int num){
    for(int row = 0; row < _gridSize ; row++){
        if(_sudokuOutput->at(row).at(col) == num)
            return true;
    }
    return false;
}

bool Sudoku::SubGridCheck(int rowStartIndex, int colStartIndex, int num){
    for(int row = 0; row < 3; row++){
        for(int col = 0; col < 3; col++){
            if(_sudokuOutput->at(row + rowStartIndex).at(col + colStartIndex) == num)
                return true;
        }
    }
    return false;
}

bool Sudoku::IsCellEmpty(int &row, int &col){
    for(row = 0; row < _gridSize; row++){
        for(col = 0; col < _gridSize; col++){
            if(_sudokuOutput->at(row).at(col) == 10)
                return true;
        }
    }
    return false;
}

bool Sudoku::IsValid(int row, int col, int num){
    return !ColumnCheck(col, num) && !RowCheck(row, num) && !SubGridCheck(row - row%3, col - col%3, num); 
}

bool Sudoku::SolveBoard(){
    int row, col;
    // Check for empty spaces, if none return true
    if(!IsCellEmpty(row, col)){
        return true;
    }
    
    // Try filling in numbers from 1 to 9 and check if its a valid placement
    for(int num = 1; num <= 9; num++){
        if(IsValid(row, col, num)){
            _sudokuOutput->at(row).at(col) = num;
            if(SolveBoard() == true)
                return true;
            _sudokuOutput->at(row).at(col) = 10;       
        }
    }
    return false;
}

void Sudoku::PrintBoard(){ //print the sudoku grid after solve
    std::cout << "\nSolution:\n\n";
    for (int row = 0; row < _gridSize; row++){
      for (int col = 0; col < _gridSize; col++){
         if(col == 3 || col == 6)
            std::cout << " | ";
         std::cout << _sudokuOutput->at(row).at(col) <<" ";
      }
      if(row == 2 || row == 5){
         std::cout << std::endl;
         for(int i = 0; i < _gridSize ; i++)
            std::cout << "---";
      }
      std::cout << std::endl;
   }
}