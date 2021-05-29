#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <initializer_list>
#include <tuple>

using std::vector;
using std::initializer_list;
using std::ostream;
using std::tuple;

class Matrix {
public:
  Matrix(initializer_list<initializer_list<double>> l);
  Matrix(size_t nrows, size_t ncols);
  double& operator()(size_t i, size_t j);
  const double& operator()(size_t i, size_t j) const;
  bool isSquare() const;
  size_t numRows() const;
  size_t numColumns() const;
  ostream& print(ostream& os = std::cout) const;
  static Matrix Identity(size_t N);
  // Matrix operations
  Matrix transpose() const;
  void swapRows(size_t i, size_t j);
  Matrix& operator+=(const Matrix& rhs);
  Matrix& operator-=(const Matrix& rhs);
  Matrix operator*(const Matrix& rhs) const;
  double diagonalSquaredSum() const;
  Matrix minor(size_t m, size_t n) const;
  // slow, not optimal
  double determinant() const;
  // LU decomposition, no pivoting
  tuple<Matrix, Matrix> LUDecomposition() const;
  // LUP decomposition
  tuple<Matrix, Matrix, Matrix> LUPDecomposition() const;
  // matrix difference
  static double rootMeanSquareError(const Matrix& matA, const Matrix& matB);
private:
  size_t m_nrows;
  size_t m_ncols;
  vector<double> m_data;
};

// Eigen solver for real symmetric matrix
class realSymmetricEigenSolver {
public:
  realSymmetricEigenSolver(const Matrix& matA, double threshold = 1e-7);
  tuple<Matrix, Matrix> solve();
private:
  Matrix m_matA;
  Matrix m_matV;
  double m_threshold;
private:
  // helper function for Eigen solver
  // calculate c and s for Jacobi rotation
  static void calc_c_s(double a_pq, double a_pp, double a_qq, double& c, double& s);
  // apply the Jacobi transformation, P^-1 * A * P
  void applyJacobiTransformation(double c, double s, size_t p, size_t q);
  // multiply the Jacobi rotation matrix, A * V (for eigenvectors)
  void multiplyJacobi(double c, double s, size_t p, size_t q);
  // A sweep of Jacobi rotations
  void JacobiSweep();
};

// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

ostream& operator<<(ostream& os, const Matrix& A);

// solve AX=B by Gaussian elimination
Matrix GaussianElimination(Matrix& matA, Matrix& matB);

// Cholesky decomposition
// matA = LL', return L
Matrix CholeskyDecomposition(const Matrix& matA);

// classical Gram-Schmidt process
tuple<Matrix, Matrix> GramSchmidtProcess(const Matrix& matA);

// modified Gram-Schmidt process
tuple<Matrix, Matrix> ModifiedGramSchmidtProcess(const Matrix& matA);

#endif // MATRIX_H
