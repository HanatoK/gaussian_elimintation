#include "Matrix.h"

// TODO: LUP decomposition
//       Singular value decomposition
//       QR decomposition

void testLinearSolver(Matrix matA, Matrix matB) {
  std::cout << "Matrix A:\n" << matA;
  std::cout << "Matrix B:\n" << matB;
  std::cout << "After Gaussian elimination:\n";
  const Matrix matX = GaussianElimination(matA, matB);
  std::cout << "Matrix A:\n" << matA;
  std::cout << "Matrix B:\n" << matB;
  std::cout << "Matrix X:\n" << matX;
}

void testEigensystem(const Matrix& matA) {
  std::cout << "Matrix A:\n" << matA;
  tuple<Matrix, Matrix> eigen = matA.realSymmetricEigenSolver();
  std::cout << "Eigenvalues:\n" << std::get<0>(eigen);
  std::cout << "Eigenvectors:\n" << std::get<1>(eigen);
}

void testDeterminant(const Matrix& matA) {
  std::cout << "Matrix A:\n" << matA;
  const double det = matA.determinant();
  std::cout << "Determinant of matA: " << det << '\n';
}

void testLU(const Matrix& matA) {
  tuple<Matrix, Matrix> lu = matA.LUDecomposition();
  std::cout << "Matrix A:\n" << matA;
  std::cout << "Matrix L:\n" << std::get<0>(lu);
  std::cout << "Matrix U:\n" << std::get<1>(lu);
  std::cout << "L*U:\n" << std::get<0>(lu)*std::get<1>(lu);
}

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

void testGramSchmidt() {
  std::cout << "void testGramSchmidt()\n";
  Matrix matA{{ 1.896457,  0.213800,  0.619222,  1.288015},
              { 0.213800,  0.039964,  0.080064,  0.142678},
              { 0.619222,  0.080064,  0.409956,  0.239832},
              { 1.288015,  0.142678,  0.239832,  1.125348}};
  tuple<Matrix, Matrix> qr = ModifiedGramSchmidtProcess(matA);
  std::cout << "Matrix A:\n" << matA;
  std::cout << "Matrix Q:\n" << std::get<0>(qr);
  std::cout << "Matrix R:\n" << std::get<1>(qr);
  std::cout << "Q*R:\n" << std::get<0>(qr)*std::get<1>(qr);
  std::cout << "Check orthogonality:\n" << std::get<0>(qr).transpose()*std::get<0>(qr);
}

int main() {
  Matrix matA{{ 2.0,  0.5,  1.0, -2.0,  3.0},
              { 0.5,  1.0,  0.1,  4.0, -9.0},
              { 1.0,  0.1, -3.0, -2.0,  0.0},
              {-2.0,  4.0, -2.0,  0.2, -1.0},
              { 3.0, -9.0,  0.0, -1.0, -0.3}};
  Matrix matB{{-2.7, -0.5},
              { 2.5, -1.9},
              { 1.2,  7.1},
              {-2.8, -4.0},
              {-0.1, -9.6}};
  testEigensystem(matA);
  testLinearSolver(matA, matB);
  testDeterminant(matA);
  testLU(matA);
  testCholesky();
  testGramSchmidt();
  return 0;
}
