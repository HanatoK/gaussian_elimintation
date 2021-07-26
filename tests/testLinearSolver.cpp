#include "Matrix.h"
#include "Spline.h"
#include "Reparametrization.h"
#include "testMatrices.h"

#include <fmt/format.h>
#include <fstream>

void testLinearSolver(Matrix matA, Matrix matB) {
  std::cout << "Matrix A:\n" << matA;
  std::cout << "Matrix B:\n" << matB;
  std::cout << "After Gaussian elimination:\n";
  const Matrix matX = GaussianElimination(matA, matB);
  std::cout << "Matrix A:\n" << matA;
  std::cout << "Matrix B:\n" << matB;
  std::cout << "Matrix X:\n" << matX;
}

int main() {
  testLinearSolver(TEST::matA, TEST::matB);
  return 0;
}
