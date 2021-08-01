#ifndef MATRIX_H
#define MATRIX_H

#include "Helper.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <initializer_list>
#include <tuple>
#include <limits>

using std::vector;
using std::initializer_list;
using std::ostream;
using std::tuple;

class Matrix {
public:
  Matrix();
  Matrix(std::istream& ifs, const std::string& delimiter = " ");
  Matrix(const vector<vector<double>>& vec);
  Matrix(initializer_list<initializer_list<double>> l);
  Matrix(size_t nrows, size_t ncols);
  double& operator()(size_t i, size_t j);
  const double& operator()(size_t i, size_t j) const;
  bool isSquare() const;
  size_t numRows() const;
  size_t numColumns() const;
  ostream& print(ostream& os = std::cout) const;
  static Matrix identity(size_t N);
  static Matrix rand(size_t N, double a = -1.0, double b = 1.0);
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
  static double rootMeanSquareError(const Matrix& matA, const Matrix& matB);
private:
  void fromVector(const vector<vector<double>>& vec);
  size_t m_nrows;
  size_t m_ncols;
  vector<double> m_data;
};

// LU Decomposition
class LUDecomposition {
public:
  LUDecomposition(Matrix matA, bool LUP = true);
  const Matrix& getL() const;
  const Matrix& getU() const;
  const Matrix& getP() const;
  const Matrix& getInverseL() const;
  const Matrix& getInverseU() const;
//   Matrix solve(const Matrix& matB) const;
  Matrix inverse() const;
private:
  void inverse_matL();
  void inverse_matU();
  bool m_LUP;
  Matrix m_matL;
  Matrix m_matU;
  Matrix m_matP;
  Matrix m_inv_matL;
  Matrix m_inv_matU;
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

// introduce zeros to column "col" of matA under row "row"
Matrix getHouseholderPLeft(const Matrix& matA, const size_t col, const size_t row);

// introduce zeros to row "row" of matA from column "col"
Matrix getHouseholderPRight(const Matrix& matA, const size_t col, const size_t row);

tuple<Matrix, Matrix> HouseholderQRNaive(const Matrix& matA);
tuple<Matrix, Matrix> HouseholderQR(const Matrix& matA);

// bidiagonalization by Householder transformation
// return P_left, X, P_right where P_left * X * P_right = matA
// and X is a bidiagonal matrix
tuple<Matrix, Matrix, Matrix> naiveBidiagonalization(Matrix matA);

// naive SVD (phase 2)
// assume matA is a real bidiagonal matrix
// return U, Σ and V
tuple<Matrix, Matrix, Matrix> SVDPhaseTwo(const Matrix& matA); 

// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
// this can be zero!!
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

// treat rows of matrix as points and calculate distances between them
vector<double> calcDistance(const Matrix& mat);

#endif // MATRIX_H
