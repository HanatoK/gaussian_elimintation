#include "Matrix.h"
#include "Spline.h"
#include "Reparametrization.h"
#include "testMatrices.h"

#include <fmt/format.h>
#include <fstream>

void testLU(const Matrix& matA) {
  LUDecomposition lu(matA, false);
  std::cout << "Matrix A:\n" << matA;
  std::cout << "Matrix L:\n" << lu.getL();
  std::cout << "Matrix U:\n" << lu.getU();
  std::cout << "L*U:\n" << lu.getL()*lu.getU();
}

void testLUP(const Matrix& matA) {
  LUDecomposition lu(matA, true);
  std::cout << "Matrix A:\n" << matA;
  std::cout << "Matrix L:\n" << lu.getL();
  std::cout << "Matrix U:\n" << lu.getU();
  std::cout << "Matrix P:\n" << lu.getP();
  std::cout << "L*U:\n" << lu.getL()*lu.getU();
  std::cout << "P*A:\n" << lu.getP()*matA;
  std::cout << "inv(U):\n" << lu.getInverseU();
  std::cout << "inv(L):\n" << lu.getInverseL();
  std::cout << "RMSE = "
            << Matrix::rootMeanSquareError(
                lu.getL()*lu.getU(), lu.getP()*matA)
            << std::endl;
}

int main() {
  testLU(TEST::matA);
  std::cout << "========== Expected NaN:\n";
  testLU(TEST::matC);
  std::cout << "====================\n";
  testLUP(TEST::matC);
  return 0;
}
