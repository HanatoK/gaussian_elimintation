#include "Matrix.h"
#include "Spline.h"

#include <fmt/format.h>

// TODO: Singular value decomposition
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

void testDeterminant(const Matrix& matA) {
  std::cout << "Matrix A:\n" << matA;
  const double det = matA.determinant();
  std::cout << "Determinant of matA: " << det << '\n';
}

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
  tuple<Matrix, Matrix> qr = GramSchmidtProcess(matA);
  std::cout << "Matrix A:\n" << matA;
  std::cout << "Matrix Q:\n" << std::get<0>(qr);
  std::cout << "Matrix R:\n" << std::get<1>(qr);
  std::cout << "Q*R:\n" << std::get<0>(qr)*std::get<1>(qr);
  std::cout << "Check orthogonality:\n" << std::get<0>(qr).transpose()*std::get<0>(qr);
}

void testModifiedGramSchmidt() {
  std::cout << "void testModifiedGramSchmidt()\n";
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

void testModifiedGramSchmidtRectangular() {
  std::cout << "void testModifiedGramSchmidtRectangular()\n";
  Matrix matA{{ 1.896457,  0.213800},
              { 0.213800,  0.039964},
              { 0.619222,  0.080064},
              { 1.288015,  0.142678}};
  tuple<Matrix, Matrix> qr = ModifiedGramSchmidtProcess(matA);
  std::cout << "Matrix A:\n" << matA;
  std::cout << "Matrix Q:\n" << std::get<0>(qr);
  std::cout << "Matrix R:\n" << std::get<1>(qr);
  std::cout << "Q*R:\n" << std::get<0>(qr)*std::get<1>(qr);
  std::cout << "Check orthogonality:\n" << std::get<0>(qr).transpose()*std::get<0>(qr);
}

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

void testInterpolateBase() {
  std::cout << "Test InterpolateBase::locate\n";
  std::vector<double> X{
    -2.0, 1.0, 2.0, 4.0, 10.0, 15.0,
    20.5, 66.3, 70.4, 127.0, 211.6, 215.2,
    266.0, 268.0, 300.6, 466,3, 512.0};
  std::vector<double> Y;
  for (const auto& a : X) {
    Y.push_back(std::sin(a));
  }
  InterpolateBase interp(X, Y, 5);
  bool index_ok = true;
  size_t i = interp.locate(80.0, &index_ok);
  std::cout << std::boolalpha << "Index = " << i
            << " (" << index_ok << ")\n";
  std::cout << "Test InterpolateBase::fastIndex\n";
  for (size_t i = 0; i < X.size(); ++i) {
    X[i] = i * 5.0;
    Y[i] = std::sin(X[i]);
  }
  interp = InterpolateBase(X, Y, 5, true);
  i = interp.fastIndex(50.0, &index_ok);
  std::cout << std::boolalpha << "Index = " << i
            << " (" << index_ok << ")\n";
}

void testSplineInterpolation() {
  const size_t N = 100;
  std::vector<double> X(N);
  std::vector<double> Y(N);
  for (size_t i = 0; i < N; ++i) {
    X[i] = i;
    Y[i] = double(i) / N * M_PI;
  }
  SplineInterpolate spline_interp(X, Y, true);
  for (size_t i = 0; i < N; ++i) {
    const double val = spline_interp.evaluate(i);
    fmt::format("x = {:6lf} ; y = {:12.7lf} ; interp = {12.7lf}\n", X[i], Y[i], val);
  }
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
  Matrix matC{{0.0, 1.2, -3.0},
              {-5.0, 0.5, 1.3},
              {2.0, 0.6, 0}};
  testEigensystem(matA);
  testLinearSolver(matA, matB);
  testDeterminant(matA);
  testLU(matA);
  testCholesky();
  testGramSchmidt();
  testModifiedGramSchmidt();
  testModifiedGramSchmidtRectangular();
  std::cout << "========== Expected NaN:\n";
  testLU(matC);
  std::cout << "====================\n";
  testLUP(matC);
  testInverse();
  testInterpolateBase();
  testSplineInterpolation();
  return 0;
}
