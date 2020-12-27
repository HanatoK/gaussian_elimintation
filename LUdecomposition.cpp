#include <cmath>
#include <algorithm>

#include "common.h"

struct LUdecomposition {
  std::vector<std::vector<double>> matA;
  std::vector<size_t> P;
  const double tol = 1e-12;
  LUdecomposition(const std::vector<std::vector<double>>& A);
};

LUdecomposition::LUdecomposition(const std::vector<std::vector<double>>& A):
matA(A), P(matA.size() + 1) {
  // modified from wikipedia
  // TODO: it seems it is not the LU decomposition I have known,
  //       need taking more time to figure out what it is.
  // number of rows or columns
  const size_t N = matA.size();
  for (size_t i = 0; i < N; ++i) P[i] = i;
  for (size_t i = 0; i < N; ++i) {
    // find the pivot
    double maxA = 0.0;
    size_t imax = i;
    for (size_t k = i; k < N; ++k) {
      double absA = std::abs(matA[k][i]);
      if (maxA < absA) {
        maxA = absA;
        imax = k;
      }
    }
    if (maxA < tol) return;
    if (imax != i) {
      // pivoting P
      std::swap(P[i], P[imax]);
      // pivoting rows of A
      std::swap(matA[i], matA[imax]);
    }
    for (size_t j = i + 1; j < N; ++j) {
      matA[j][i] /= matA[i][i];
      for (size_t k = i + 1; k < N; ++k) {
        matA[j][k] -= matA[j][i] * matA[i][k];
      }
    }
  }
}

int main() {
  std::vector<std::vector<double>> A = {{1,2,4},
                                        {3,8,14},
                                        {2,6,13}};
  LUdecomposition d(A);
  std::cout << "LU:\n";
  PrintMatrix(d.matA);
}
