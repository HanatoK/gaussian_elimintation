#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <string>
#include <chrono>

using std::vector;
using std::swap;

void PrintMatrix(const vector<vector<double>>& mat, std::ostream& os = std::cout) {
  std::ios_base::fmtflags f(os.flags());
  os << std::fixed;
  os << std::setprecision(7);
  for (const auto& v: mat) {
    for (const auto& item : v) {
      os << std::setw(12) << item << " ";
    }
    os << '\n';
  }
  os.flags(f);
}

vector<vector<double>> GaussianElimination(vector<vector<double>>& matA, vector<vector<double>>& matB) {
  // assume matA is always a square matrix
  const size_t N = matA.size();
  // B has M columns
  const size_t M = matB[0].size();
  // iterate over columns
  for (size_t j = 0; j < N; ++j) {
    // iterate over rows and find the pivot
    double pivot = matA[j][j];
    size_t pivot_index = j;
    for (size_t k = j; k < N; ++k) {
      if (abs(matA[k][j]) > abs(pivot)) {
        pivot = matA[k][j];
        pivot_index = k;
      }
    }
    // check if the pivot is zero
    if (pivot == 0.0) continue;
    // current row is also j
    // swap row j and row pivot_index
    if (j != pivot_index) {
      swap(matA[j], matA[pivot_index]);
      swap(matB[j], matB[pivot_index]);
    }
    // subtract the successive rows from the current row
    for (size_t k = j + 1; k < N; ++k) {
      const double factor = matA[k][j] / pivot;
      for (size_t i = j; i < N; ++i) {
        matA[k][i] = matA[k][i] - matA[j][i] * factor;
      }
      for (size_t i = 0; i < M; ++i) {
        matB[k][i] = matB[k][i] - matB[j][i] * factor;
      }
    }
  }
  // solve X, backsubstitution
  vector<vector<double>> matX(N, vector<double>(M));
  // boundary check
  if (N > 0) {
    for (int j = N - 1; j >= 0; --j) {
      for (size_t i = 0; i < M; ++i) {
        if (j == N - 1) {
          matX[j][i] = matB[j][i] / matA[j][j];
        } else {
          double sum = 0.0;
          for (int k = 1; k < N - j; ++k) {
            sum += matX[j+k][i] * matA[j][j+k];
          }
          matX[j][i] = (matB[j][i] - sum) / matA[j][j];
        }
      }
    }
  }
  return matX;
}

vector<vector<double>> GaussianElimination2(vector<vector<double>>& matA, vector<vector<double>>& matB) {
  // assume matA is always a square matrix
  const size_t N = matA.size();
  // B has M columns
  const size_t M = matB[0].size();
  // row index of column pivots
  vector<size_t> pivot_indices(N, 0);
  // bookkeep the used rows
  vector<bool> used_rows(N, false);
  // iterate over columns
  for (size_t j = 0; j < N; ++j) {
    // iterate over rows and find the pivot
    bool firsttime = true;
    double pivot = 0.0;
    for (size_t k = 0; k < N; ++k) {
      if (used_rows[k] == false) {
        // find column pivot in the remaining rows
        if (firsttime) {
          pivot = matA[k][j];
          pivot_indices[j] = k;
          firsttime = false;
        } else {
          if (abs(matA[k][j]) > abs(pivot)) {
            pivot = matA[k][j];
            pivot_indices[j] = k;
          }
        }
      }
    }
    used_rows[pivot_indices[j]] = true;
    // check if the pivot is zero
    if (pivot == 0.0) continue;
    for (size_t k = 0; k < N; ++k) {
      if (used_rows[k] == false) {
        const double factor = matA[k][j] / pivot;
#ifdef DEBUG
        std::cout << "k = " << k << " ; factor = " << factor << std::endl;
#endif
        for (size_t i = j; i < N; ++i) {
          matA[k][i] = matA[k][i] - matA[pivot_indices[j]][i] * factor;
        }
        for (size_t i = 0; i < M; ++i) {
          matB[k][i] = matB[k][i] - matB[pivot_indices[j]][i] * factor;
        }
      }
    }
#ifdef DEBUG
    std::cout << "Matrix A:\n";
    PrintMatrix(matA);
#endif
  }
#ifdef DEBUG
  std::cout << "pivot_indices:\n";
  for (const auto& i : pivot_indices) {
    std::cout << i << std::endl;
  }
#endif
  // solve X, backsubstitution
  vector<vector<double>> matX(N, vector<double>(M));
  // boundary check
  if (N > 0) {
    for (int j = N - 1; j >= 0; --j) {
      // first, we need to find which row has the last pivot
      const size_t l = pivot_indices[j];
      for (size_t i = 0; i < M; ++i) {
        if (j == N - 1) {
          matX[j][i] = matB[l][i] / matA[l][j];
        } else if (j == N - 2) {
          matX[j][i] = (matB[l][i] - matX[j+1][i] * matA[l][j+1]) / matA[l][j];
        } else {
          double sum = 0.0;
          for (int k = 1; k < N - j; ++k) {
            sum += matX[j+k][i] * matA[l][j+k];
          }
          matX[j][i] = (matB[pivot_indices[j]][i] - sum) / matA[l][j];
        }
      }
    }
  }
  return matX;
}

void splitString(const std::string& data, const std::string& delim, std::vector<double>& dest) {
    if (delim.empty()) {
        dest.push_back(std::stod(data));
        return;
    }
    size_t index = 0, new_index = 0;
    std::string tmpstr;
    while (index != data.length()) {
        new_index = data.find(delim, index);
        if (new_index != std::string::npos) tmpstr = data.substr(index, new_index - index);
        else tmpstr = data.substr(index, data.length());
        if (!tmpstr.empty()) {
            dest.push_back(std::stod(tmpstr));
        }
        if (new_index == std::string::npos) break;
        index = new_index + 1;
    }
}

vector<vector<double>> readTo2DVector(const std::string& filename) {
  std::ifstream ifs(filename);
  std::string line;
  vector<vector<double>> mat;
  while (std::getline(ifs, line)) {
    vector<double> fields;
    splitString(line, " ", fields);
    mat.push_back(std::move(fields));
  }
  return mat;
}

int main() {
  vector<vector<double>> A = readTo2DVector("matA.txt");
  vector<vector<double>> B = readTo2DVector("matB.txt");
  auto start1 = std::chrono::high_resolution_clock::now();
  const auto X = GaussianElimination(A, B);
  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end1 - start1;
  std::cout << "With swap: " << diff.count() << " s.\n";
  std::ofstream ofs("solution.txt");
  PrintMatrix(X, ofs);
  auto A2 = A;
  auto B2 = B;
  auto start2 = std::chrono::high_resolution_clock::now();
  const auto X2 = GaussianElimination2(A2, B2);
  auto end2 = std::chrono::high_resolution_clock::now();
  diff = end2 - start2;
  std::cout << "Without swap: " << diff.count() << " s.\n";
  std::ofstream ofs2("solution2.txt");
  PrintMatrix(X2, ofs2);
  return 0;
}
