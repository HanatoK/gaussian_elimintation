#include <cmath>
#include <algorithm>

#include "common.h"

using std::swap;

struct LUdecomposition {
  std::vector<std::vector<double>> matL;
  std::vector<std::vector<double>> matU;
  std::vector<std::vector<double>> matA;
  std::vector<size_t> P;
  const double tol = 1e-12;
  LUdecomposition(const std::vector<std::vector<double>>& A);
};

LUdecomposition::LUdecomposition(const std::vector<std::vector<double>>& A):
matL(A.size(), std::vector<double>(A.size(), 0.0)), matU(matL), matA(A), P(A.size(), 0) {
  const int N = matA.size();
  for (size_t i = 0; i < N; ++i) matL[i][i] = 1;
// #define LU_JIK
#ifdef LU_JIK
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < j + 1; ++i) {
      double sum = 0.0;
      for (size_t k = 0; k < i; ++k) {
        sum += matL[i][k] * matU[k][j];
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
#else
  // copy from NR
  // TODO: figure out what happens
  double big = 0.0;
  size_t imax = 0;
  double d = 1.0;
  double tmp;
  std::vector<double> scaling(N);
  auto lu = matA;
  for (size_t i = 0; i < N; ++i) {
    big = 0.0;
    for (size_t j = 0; j < N; ++j) {
      tmp = std::abs(lu[i][j]);
      if (tmp > big) {
        big = tmp;
      }
    }
    // throw if big == 0
    scaling[i] = 1.0 / big;
  }
  for (size_t k = 0; k < N; ++k) {
    for (size_t i = k; i < N; ++i) {
      tmp = scaling[i] * std::abs(lu[i][k]);
      if (tmp > big) {
        big = tmp;
        imax = i;
      }
    }
    if (k != imax) {
      // interchange rows
      swap(lu[imax], lu[k]);
      d = -d;
      scaling[imax] = scaling[k];
    }
    P[k] = imax;
    for (size_t i = k + 1; i < N; ++i) {
      // TODO
      tmp = lu[i][k] /= lu[k][k];
      for (size_t j = k + 1; j < N; ++j) {
        lu[i][j] -= tmp * lu[k][j];
      }
    }
  }
  // copy back to matL and matU
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (i > j) matL[i][j] = lu[i][j];
      else matU[i][j] = lu[i][j];
    }
  }
#endif
}

int main() {
  std::vector<std::vector<double>> A = {{3,2,4},
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
