#include <iostream>
#include <vector>

class Board {
public:
    Board() : grid(9, std::vector<int>(9, 0)) {}

    bool isComplete() const {
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (grid[i][j] == 0) return false;
            }
        }
        return true;
    }

    bool isEmpty(int row, int col) const {
        return grid[row][col] == 0;
    }

    bool isValid(int row, int col, int num) const {
        // Check row and column
        for (int i = 0; i < 9; ++i) {
            if (grid[row][i] == num || grid[i][col] == num) return false;
        }
        // Check 3x3 block
        int startRow = (row / 3) * 3;
        int startCol = (col / 3) * 3;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (grid[startRow + i][startCol + j] == num) return false;
            }
        }
        return true;
    }

    void place(int row, int col, int num) {
        grid[row][col] = num;
    }

    void remove(int row, int col) {
        grid[row][col] = 0;
    }

    void print() const {
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                std::cout << grid[i][j] << " ";
            }
            std::cout << "\n";
        }
    }

    void setInitial(const std::vector<std::vector<int>>& initial) {
        grid = initial;
    }

private:
    std::vector<std::vector<int>> grid;
};

// Recursive backtracking Sudoku solver
bool solveSudoku(Board &b) {
    if (b.isComplete()) return true;
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            if (b.isEmpty(i, j)) {
                for (int n = 1; n <= 9; ++n) {
                    if (b.isValid(i, j, n)) {
                        b.place(i, j, n);
                        if (solveSudoku(b)) {
                            return true;
                        }
                        b.remove(i, j);
                    }
                }
                return false; // no valid number found
            }
        }
    }
    return true;
}

int main() {
    Board board;
    std::vector<std::vector<int>> initial = {
        {5,3,0,0,7,0,0,0,0},
        {6,0,0,1,9,5,0,0,0},
        {0,9,8,0,0,0,0,6,0},
        {8,0,0,0,6,0,0,0,3},
        {4,0,0,8,0,3,0,0,1},
        {7,0,0,0,2,0,0,0,6},
        {0,6,0,0,0,0,2,8,0},
        {0,0,0,4,1,9,0,0,5},
        {0,0,0,0,8,0,0,7,9}
    };
    board.setInitial(initial);

    std::cout << "Initial Sudoku:\n";
    board.print();

    if (solveSudoku(board)) {
        std::cout << "\nSolved Sudoku:\n";
        board.print();
    } else {
        std::cout << "\nNo solution exists.\n";
    }

    return 0;
}
