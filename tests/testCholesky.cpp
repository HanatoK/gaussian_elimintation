#include "Matrix.h"
#include "Spline.h"
#include "Reparametrization.h"
#include "testMatrices.h"

#include <fmt/format.h>
#include <fstream>

void testCholesky() {
  Matrix matA{{ 1.896457,  0.213800,  0.619222,  1.288015},
              { 0.213800,  0.039964,  0.080064,  0.142678},
              { 0.619222,  0.080064,  0.409956,  0.239832},
              { 1.288015,  0.142678,  0.239832,  1.125348}};
  const Matrix L = CholeskyDecomposition(matA);
  std::cout << "Matrix A:\n" << matA;
  std::cout << "Matrix L:\n" << L;
  std::cout << "L*L':\n" << L * L.transpose();
}

int main() {
  testCholesky();
  return 0;
}
