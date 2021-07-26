#include "Matrix.h"
#include "Spline.h"
#include "Reparametrization.h"
#include "testMatrices.h"

#include <fmt/format.h>
#include <fstream>

void testEigensystem(const Matrix& matA) {
  std::cout << "Matrix A:\n" << matA;
  realSymmetricEigenSolver solver(matA);
  tuple<Matrix, Matrix> eigen = solver.solve();
  const Matrix& Lambda = std::get<0>(eigen);
  const Matrix& Q = std::get<1>(eigen);
  std::cout << "Eigenvalues:\n" << Lambda;
  std::cout << "Eigenvectors:\n" << Q;
  const Matrix tmp = Q * Lambda * Q.transpose();
  std::cout << "Q*Λ*Q'\n" << tmp;
  std::cout << "RMSE = " << Matrix::rootMeanSquareError(tmp, matA) << std::endl;
}

int main() {
  testEigensystem(TEST::matA);
  return 0;
}
