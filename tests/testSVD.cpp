#include "Matrix.h"
#include "Spline.h"
#include "Reparametrization.h"
#include "testMatrices.h"

#include <fmt/format.h>
#include <fstream>

void testSVDPhaseOne(const Matrix& matA) {
  auto stage1 = naiveBidiagonlization(matA);
  const auto P_left = std::get<0>(stage1);
  const auto R = std::get<1>(stage1);
  const auto P_right = std::get<2>(stage1);
  std::cout << "P_left:\n" << P_left;
  std::cout << "R:\n" << R;
  std::cout << "P_right:\n" << P_right;
  std::cout << "Check orthogonality of P_left:\n"
            << P_left * P_left.transpose();
  std::cout << "Check orthogonality of P_right:\n"
            << P_right * P_right.transpose();
  const auto result = P_left * R * P_right;
  std::cout << "P_left * R * P_right:\n";
  std::cout << result;
  std::cout << "RMSE = "
            << Matrix::rootMeanSquareError(result, matA)
            << std::endl;
}

int main() {
  std::cout << "SVD phase 1: bidiagonlization\n";
  testSVDPhaseOne(TEST::matA);
  return 0;
}
