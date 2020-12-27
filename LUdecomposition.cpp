#include <cmath>
#include <algorithm>

#include "common.h"

struct LUdecomposition {
  std::vector<std::vector<double>> matL;
  std::vector<std::vector<double>> matU;
  std::vector<std::vector<double>> matA;
  std::vector<size_t> P;
  const double tol = 1e-12;
  LUdecomposition(const std::vector<std::vector<double>>& A);
};

LUdecomposition::LUdecomposition(const std::vector<std::vector<double>>& A):
matL(A.size(), std::vector<double>(A.size(), 0.0)), matU(matL), matA(A) {
  const int N = matA.size();
  for (size_t i = 0; i < N; ++i) matL[i][i] = 1;
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < j + 1; ++i) {
      double sum = 0.0;
      if (i > 0) {
        for (size_t k = 0; k < i; ++k) {
          sum += matL[i][k] * matU[k][j];
        }
      }
      matU[i][j] = matA[i][j] - sum;
    }
    for (size_t i = j + 1; i < N; ++i) {
      double sum = 0.0;
      for (size_t k = 0; k < j; ++k) {
        sum += matL[i][k] * matU[k][j];
      }
      matL[i][j] = (matA[i][j] - sum) / matU[j][j];
    }
  }
}

int main() {
  std::vector<std::vector<double>> A = {{1,2,4},
                                        {3,8,14},
                                        {2,6,1}};
  LUdecomposition d(A);
  std::cout << "A:\n";
  PrintMatrix(A);
  std::cout << "L:\n";
  PrintMatrix(d.matL);
  std::cout << "U:\n";
  PrintMatrix(d.matU);
}
