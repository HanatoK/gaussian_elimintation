#include "Matrix.h"
#include "Spline.h"
#include "Reparametrization.h"
#include "testMatrices.h"

#include <fmt/format.h>
#include <fstream>

Matrix testSVDPhaseOne(const Matrix& matA) {
  auto stage1 = naiveBidiagonalization(matA);
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
  return R;
}

void testSVDPhaseTwo(const Matrix& matA) {
  auto stage2 = SVDPhaseTwo(matA);
  const auto P_left = std::get<0>(stage2);
  const auto R = std::get<1>(stage2);
  const auto P_right = std::get<2>(stage2);
  std::cout << "U:\n" << P_left;
  std::cout << "sigma:\n" << R;
  std::cout << "V':\n" << P_right;
  std::cout << "Check orthogonality of U:\n"
            << P_left * P_left.transpose();
  std::cout << "Check orthogonality of V':\n"
            << P_right * P_right.transpose();
  const auto result = P_left * R * P_right;
  std::cout << "U * sigma * V':\n";
  std::cout << result;
  std::cout << "RMSE = "
            << Matrix::rootMeanSquareError(result, matA)
            << std::endl;
}

void testSVD(const Matrix& matA) {
  std::cout << "Test SVD:\n";
  const auto Y = SVD(matA);
  const auto U = std::get<0>(Y);
  const auto sigma = std::get<1>(Y);
  const auto V = std::get<2>(Y);
  std::cout << "U = \n" << U;
  std::cout << "sigma = \n" << sigma;
  std::cout << "V' = \n" << V;
  std::cout << "Check orthogonality of U:\n" << U * U.transpose();
  std::cout << "Check orthogonality of V':\n" << V * V.transpose();
  const auto result = U * sigma * V;
  std::cout << "U*sigma*V'\n" << U * sigma * V;
  std::cout << "RMSE = "
            << Matrix::rootMeanSquareError(result, matA)
            << std::endl;
}

int main() {
//   {
//     std::cout << "Matrix D:\n" << TEST::matD;
//     std::cout << "SVD phase 1: bidiagonalization\n";
//     const auto R = testSVDPhaseOne(TEST::matD);
//     std::cout << "SVD phase 2: eigendecomposition\n";
//     testSVDPhaseTwo(R);
//   }
//   {
//     std::cout << "Matrix A:\n" << TEST::matA;
//     std::cout << "SVD phase 1: bidiagonalization\n";
//     const auto R = testSVDPhaseOne(TEST::matA);
//     std::cout << "SVD phase 2: eigendecomposition\n";
//     testSVDPhaseTwo(R);
//   }
  {
    std::cout << "Matrix D:\n" << TEST::matD;
    testSVD(TEST::matD);
    std::cout << "Matrix A:\n" << TEST::matA;
    testSVD(TEST::matA);
    std::cout << "Matrix D':\n" << TEST::matD.transpose();
    testSVD(TEST::matD.transpose());
  }
  return 0;
}
