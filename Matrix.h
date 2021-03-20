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
  void transpose();
  void swapRows(size_t i, size_t j);
  Matrix& operator+=(const Matrix& rhs);
  Matrix& operator-=(const Matrix& rhs);
  Matrix operator*(const Matrix& rhs);
  double diagonalSquaredSum() const;
  Matrix minor(size_t m, size_t n) const;
  // slow, not optimal
  double determinant() const;
  // Eigen solver for real symmetric matrix
  tuple<Matrix, Matrix> realSymmetricEigenSolver(double threshold = 1e-7) const;
  // LU decomposition, no pivoting
  tuple<Matrix, Matrix> LUDecomposition() const;
private:
  size_t m_nrows;
  size_t m_ncols;
  vector<double> m_data;
private:
  // helper function for Eigen solver
  // calculate c and s for Jacobi rotation
  void calc_c_s(double a_pq, double a_pp, double a_qq, double& c, double& s);
  // apply the Jacobi transformation, P^-1 * A * P
  void applyJacobiTransformation(double c, double s, size_t p, size_t q);
  // multiply the Jacobi rotation matrix, A * V (for eigenvectors)
  void multiplyJacobi(double c, double s, size_t p, size_t q);
  // A sweep of Jacobi rotations
  static void JacobiSweep(Matrix& matA, Matrix& matP);
};

// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

ostream& operator<<(ostream& os, const Matrix& A);

// solve AX=B by Gaussian elimination
Matrix GaussianElimination(Matrix& matA, Matrix& matB);
