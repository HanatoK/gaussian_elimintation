#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>

using std::vector;
using std::swap;

void PrintMatrix(const vector<vector<double>>& mat) {
  using std::cout;
  std::ios_base::fmtflags f(cout.flags());
  cout << std::fixed;
  cout << std::setprecision(10);
  for (const auto& v: mat) {
    for (const auto& item : v) {
      cout << std::setw(15) << item << " ";
    }
    cout << '\n';
  }
  cout.flags(f);
}

vector<vector<double>> GaussianElimination(vector<vector<double>>& matA, vector<vector<double>>& matB) {
  // assume matA is always a square matrix
  const size_t N = matA.size();
  // B has M columns
  const size_t M = matB[0].size();
  // iterate over columns
  for (size_t j = 0; j < N; ++j) {
#ifdef DEBUG
    std::cout << "Col: " << j << "\n";
    std::cout << "A:\n";
    PrintMatrix(matA);
    std::cout << "B:\n";
    PrintMatrix(matB);
#endif
    // iterate over rows and find the pivot
    double pivot = matA[j][j];
    size_t pivot_index = j;
    for (size_t k = j; k < N; ++k) {
      if (abs(matA[k][j]) > abs(pivot)) {
        pivot = matA[k][j];
        pivot_index = k;
      }
    }
    // current row is also j
    // swap row j and row pivot_index
    if (j != pivot_index) {
      swap(matA[j], matA[pivot_index]);
      swap(matB[j], matB[pivot_index]);
    }
    // current row is also j
    // divide the pivot
    for (size_t i = j; i < N; ++i) {
      matA[j][i] /= pivot;
    }
    for (size_t i = 0; i < M; ++i) {
      matB[j][i] /= pivot;
    }
#ifdef DEBUG
    std::cout << "Col: " << j << "\n";
    std::cout << "A:\n";
    PrintMatrix(matA);
    std::cout << "B:\n";
    PrintMatrix(matB);
#endif
    // subtract the successive rows from the current row
    for (size_t k = j + 1; k < N; ++k) {
      const double factor = matA[k][j];
      for (size_t i = j; i < N; ++i) {
        matA[k][i] = matA[k][i] - matA[j][i] * factor;
      }
      for (size_t i = 0; i < M; ++i) {
        matB[k][i] = matB[k][i] - matB[j][i] * factor;
      }
    }
  }
  // solve X
  vector<vector<double>> matX(N, vector<double>(M));
  // boundary check
  if (N > 0) {
    for (int j = N - 1; j >= 0; --j) {
      for (size_t i = 0; i < M; ++i) {
        if (j == N - 1) {
          matX[j][i] = matB[j][i];
        }/* else if (j == N - 2) {
          matX[j][i] = matB[j][i] - matX[j+1][i] * matA[j][j+1];
        } else if (j == N - 3) {
          matX[j][i] = matB[j][i] - matX[j+1][i] * matA[j][j+1] - matX[j+2][i] * matA[j][j+2];
        } else if (j == N - 4) {
          matX[j][i] = matB[j][i] - matX[j+1][i] * matA[j][j+1] - matX[j+2][i] * matA[j][j+2] - matX[j+3][i] * matA[j][j+3];
        }*/ else {
          double sum = 0.0;
          for (int k = 1; k < N - j; ++k) {
            sum += matX[j+k][i] * matA[j][j+k];
          }
          matX[j][i] = matB[j][i] - sum;
        }
      }
    }
  }
  return matX;
}

int main() {
  vector<vector<double>> A{{2, 1, 3, -1, 0},
                           {4, 5, -2, 2, -1},
                           {-1, 1, 2, 3, -1},
                           {3, -4, 2, 1, -0.5},
                           {1, 2, -1, 0, -4}};
  vector<vector<double>> B{{1, 3},
                           {2, 0},
                           {-3, -2},
                           {2, -3},
                           {0, -4}};
  const auto X = GaussianElimination(A, B);
  std::cout << "Matrix A:\n";
  PrintMatrix(A);
  std::cout << "Matrix B:\n";
  PrintMatrix(B);
  std::cout << "Solution X:\n";
  PrintMatrix(X);
  return 0;
}
