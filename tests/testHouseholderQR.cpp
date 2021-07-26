#include "Matrix.h"
#include "Spline.h"
#include "Reparametrization.h"
#include "testMatrices.h"

#include <fmt/format.h>
#include <fstream>

void testHouseholderQR(const Matrix& matA) {
  tuple<Matrix, Matrix> qr = HouseholderQR(matA);
  std::cout << "Matrix A:\n" << matA;
  std::cout << "Matrix Q:\n" << std::get<0>(qr);
  std::cout << "Matrix R:\n" << std::get<1>(qr);
  std::cout << "Q*R:\n" << std::get<0>(qr)*std::get<1>(qr);
  std::cout << "Check orthogonality:\n" << std::get<0>(qr).transpose()*std::get<0>(qr);
  const auto result = std::get<0>(qr)*std::get<1>(qr);
  std::cout << "RMSE = "
            << Matrix::rootMeanSquareError(result, matA)
            << std::endl;
}

int main() {
  testHouseholderQR(TEST::matA);
  testHouseholderQR(TEST::matB);
  return 0;
}
