#include "Matrix.h"
#include "Spline.h"
#include "Reparametrization.h"
#include "testMatrices.h"

#include <fmt/format.h>
#include <fstream>

void testInverse() {
  std::cout << "void testInverse()\n";
  Matrix matA = Matrix::rand(5);
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
  std::cout << "inv(A):\n" << lu.inverse();
  std::cout << "A*inv(A)=\n" << matA*lu.inverse();
}

int main() {
  testInverse();
  return 0;
}
