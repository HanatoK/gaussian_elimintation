#include "Matrix.h"
#include "Spline.h"
#include "Reparametrization.h"
#include "testMatrices.h"

#include <fmt/format.h>
#include <fstream>

void testQREigen(const Matrix& matA) {
  Matrix mat_tmp = matA;
  auto eigen_vecs = Matrix::identity(matA.numColumns());
  for (size_t i = 0; i < 200; ++i) {
    tuple<Matrix, Matrix> qr = HouseholderQR(mat_tmp);
    const auto lu_decomp = LUDecomposition(std::get<0>(qr));
    const auto q_inv = lu_decomp.inverse();
    eigen_vecs = eigen_vecs * std::get<0>(qr);
    mat_tmp = q_inv * mat_tmp * std::get<0>(qr);
  }
  std::cout << mat_tmp << std::endl;
  std::cout << eigen_vecs << std::endl;
}

int main() {
  testQREigen(TEST::matA);
  return 0;
}
