#include "Matrix.h"

#include <stdexcept>
#include <fmt/format.h>

Matrix::Matrix(initializer_list<initializer_list<double>> l) {
  m_nrows = l.size();
  m_ncols = l.begin()->size();
  m_data.resize(m_nrows * m_ncols);
  size_t index = 0;
  for (const auto& r: l) {
    if (r.size() != m_ncols) {
      throw std::invalid_argument("Rows have different numbers of elements.");
    }
    for (const auto& elem: r) {
      m_data[index++] = elem;
    }
  }
}

Matrix::Matrix(size_t nrows, size_t ncols) {
  m_nrows = nrows;
  m_ncols = ncols;
  m_data.assign(m_nrows * m_ncols, 0);
}

double& Matrix::operator()(size_t i, size_t j) {
  return m_data[i * m_ncols + j];
}

const double& Matrix::operator()(size_t i, size_t j) const {
  return m_data[i * m_ncols + j];
}

bool Matrix::isSquare() const {
  return m_ncols == m_nrows;
}

size_t Matrix::numRows() const {
  return m_nrows;
}

size_t Matrix::numColumns() const {
  return m_ncols;
}

ostream& Matrix::print(ostream& os) const {
  for (size_t i = 0; i < m_nrows; ++i) {
    for (size_t j = 0; j < m_ncols; ++j) {
      os << fmt::format(" {:12.7f}", (*this)(i, j));
    }
    os << '\n';
  }
  return os;
}

Matrix Matrix::Identity(size_t N) {
  Matrix result(N, N);
  for (size_t i = 0; i < N; ++i) {
    result(i, i) = 1.0;
  }
  return result;
}

Matrix Matrix::transpose() const {
  Matrix result(this->numColumns(), this->numRows());
  for (size_t i = 0; i < numRows(); ++i) {
    for (size_t j = 0; j < numColumns(); ++j) {
//       std::swap((*this)(i, j), (*this)(j, i));
      result(j, i) = (*this)(i, j);
    }
  }
  return result;
}

void Matrix::swapRows(size_t i, size_t j) {
  for (size_t k = 0; k < m_ncols; ++k) {
    std::swap((*this)(i, k), (*this)(j, k));
  }
}

ostream& operator<<(ostream& os, const Matrix& A) {
  return A.print(os);
}

Matrix& Matrix::operator+=(const Matrix& rhs) {
  if ((this->numRows() != rhs.numRows()) ||
      (this->numColumns() != rhs.numColumns())) {
    throw std::invalid_argument("The dimensions of matrices mismatch.");
  }
  for (size_t i = 0; i < m_data.size(); ++i) {
    m_data[i] += rhs.m_data[i];
  }
  return *this;
}

Matrix& Matrix::operator-=(const Matrix& rhs) {
  if ((this->numRows() != rhs.numRows()) ||
      (this->numColumns() != rhs.numColumns())) {
    throw std::invalid_argument("The dimensions of matrices mismatch.");
  }
  for (size_t i = 0; i < m_data.size(); ++i) {
    m_data[i] -= rhs.m_data[i];
  }
  return *this;
}

Matrix Matrix::operator*(const Matrix& rhs) const {
  if (this->numColumns() != rhs.numRows()) {
    throw std::invalid_argument(
      "The number of columns of the left-hand-side matrix does not match the "
      "number of rows of the right-hand-side one in matrix multiplication.");
  }
  const size_t new_rows = this->numRows();
  const size_t new_cols = rhs.numColumns();
  Matrix result(new_rows, new_cols);
  for (size_t i = 0; i < new_rows; ++i) {
    for (size_t j = 0; j < new_cols; ++j) {
      for (size_t k = 0; k < this->numColumns(); ++k) {
        result(i, j) += (*this)(i, k) * rhs(k,j);
      }
    }
  }
  return result;
}

tuple<Matrix, Matrix> Matrix::realSymmetricEigenSolver(double threshold) const {
  Matrix matA(*this);
  Matrix matV = Matrix::Identity(numRows());
  const double num_diagonal_elements = numRows() * (numRows() - 1) / 2.0;
  double off_diag_sum = std::sqrt(matA.diagonalSquaredSum() / num_diagonal_elements);
  // is this enough?
  const size_t max_iteration = 20;
  size_t iteration = 0;
  while (off_diag_sum > threshold) {
    Matrix::JacobiSweep(matA, matV);
    off_diag_sum = std::sqrt(matA.diagonalSquaredSum() / num_diagonal_elements);
    ++iteration;
    std::cout << fmt::format("Sweep {:02d}: off-diagonal squared sum = {:8.4f}\n", iteration, off_diag_sum);
    if (iteration >= max_iteration) {
      std::cerr << "Maximum iterations reached in Eigen solver!\n";
      break;
    }
  }
  return std::make_tuple(matA, matV);
}

double Matrix::diagonalSquaredSum() const {
  double sum = 0.0;
  for (size_t i = 0; i < numRows(); ++i) {
    for (size_t j = i + 1; j < numColumns(); ++j) {
      sum += (*this)(i, j) * (*this)(i, j);
    }
  }
  return sum;
}

Matrix Matrix::minor(size_t m, size_t n) const {
  Matrix result(numRows() - 1, numColumns() - 1);
  for (size_t i = 0; i < numRows(); ++i) {
    for (size_t j = 0; j < numColumns(); ++j) {
      if (i < m && j < n) result(i, j) = (*this)(i, j);
      if (i > m && j < n) result(i-1, j) = (*this)(i, j);
      if (i < m && j > n) result(i, j-1) = (*this)(i, j);
      if (i > m && j > n) result(i-1, j-1) = (*this)(i, j);
    }
  }
  return result;
}

double Matrix::determinant() const {
  if (!isSquare()) {
    throw std::invalid_argument("Determinant is only available for square matrix.");
  }
  if (numRows() == 2) {
    return m_data[0] * m_data[3] - m_data[1] * m_data[2];
  } else {
    // this is slow and not the typical numerical way...
    double result = 0;
    for (size_t j = 0; j < numColumns(); ++j) {
      const Matrix m = this->minor(0, j);
      result += std::pow(-1.0, j+0) * (*this)(0, j) * m.determinant();
    }
    return result;
  }
}

tuple<Matrix, Matrix> Matrix::LUDecomposition() const {
  if (!isSquare()) {
    throw std::invalid_argument("LUDecomposition is only available for square matrix.");
  }
  const size_t N = numRows();
  Matrix matL(N, N);
  Matrix matU(N, N);
  for (size_t i = 0; i < N; ++i) matL(i, i) = 1.0;
  // no pivoting!!
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < j + 1; ++i) {
      double sum = 0.0;
      for (size_t k = 0; k < i; ++k) {
        sum += matL(i, k) * matU(k, j);
      }
      matU(i, j) = (*this)(i, j) - sum;
    }
    for (size_t i = j + 1; i < N; ++i) {
      double sum = 0.0;
      for (size_t k = 0; k < j; ++k) {
        sum += matL(i, k) * matU(k, j);
      }
      matL(i, j) = ((*this)(i, j) - sum) / matU(j, j);
    }
  }
  return std::make_tuple(matL, matU);
}

void Matrix::calc_c_s(double a_pq, double a_pp, double a_qq,
                      double& c, double& s) {
  const double theta = 0.5 * (a_qq - a_pp) / a_pq;
  const double sign = sgn(theta);
  const double t = sign / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
  c = 1.0 / std::sqrt(t * t + 1.0);
  s = t * c;
}

void Matrix::applyJacobiTransformation(double c, double s, size_t p, size_t q) {
  const double c2 = c*c;
  const double s2 = s*s;
  const double cs = c*s;
  // is it possible to optimize the copy?
  const Matrix old_matrix(*this);
  for (size_t i = 0; i < numRows(); ++i) {
    for (size_t j = 0; j < numColumns(); ++j) {
      if (i != p && i != q) {
        if (j == p) {
          (*this)(i, p) = c * old_matrix(i, p) - s * old_matrix(i, q);
          (*this)(p, i) = (*this)(i, p);
        }
        if (j == q) {
          (*this)(i, q) = c * old_matrix(i, q) + s * old_matrix(i, p);
          (*this)(q, i) = (*this)(i, q);
        }
      }
    }
  }
  (*this)(p, p) = c2 * old_matrix(p, p) + s2 * old_matrix(q, q) -
                  2.0 * cs * old_matrix(p, q);
  (*this)(q, q) = s2 * old_matrix(p, p) + c2 * old_matrix(q, q) +
                  2.0 * cs * old_matrix(p, q);
  (*this)(p, q) = 0;
  (*this)(q, p) = (*this)(p, q);
}

void Matrix::multiplyJacobi(double c, double s, size_t p, size_t q) {
  // is it possible to optimize the copy?
  const Matrix old_matrix(*this);
  for (size_t i = 0; i < numRows(); ++i) {
    for (size_t j = 0; j < numColumns(); ++j) {
      if (j == p) (*this)(i, p) = c * old_matrix(i, p) - s * old_matrix(i, q);
      if (j == q) (*this)(i, q) = s * old_matrix(i, p) + c * old_matrix(i, q);
    }
  }
}

void Matrix::JacobiSweep(Matrix& matA, Matrix& matP) {
  const size_t nrows = matA.numRows();
  const size_t ncols = matA.numColumns();
  double c, s;
  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = i + 1; j < ncols; ++j) {
      const double a_pq = matA(i, j);
      const double a_pp = matA(i, i);
      const double a_qq = matA(j, j);
      if (std::abs(a_pq) > 0) {
        matA.calc_c_s(a_pq, a_pp, a_qq, c, s);
        matA.applyJacobiTransformation(c, s, i, j);
        matP.multiplyJacobi(c, s, i, j);
      }
    }
  }
}

Matrix GaussianElimination(Matrix& matA, Matrix& matB) {
  // assume matA is always a square matrix
  const size_t N = matA.numRows();
  if (!matA.isSquare()) {
    throw std::invalid_argument("Gaussian elimination is only implemented for square matrices.");
  }
  // B has M columns
  const size_t M = matB.numColumns();
  // row index of column pivots
  vector<size_t> pivot_indices(N, 0);
  // bookkeep the used rows
  vector<bool> used_rows(N, false);
  // iterate over columns
  for (size_t j = 0; j < N; ++j) {
    // iterate over rows and find the pivot
    bool firsttime = true;
    double pivot = 0.0;
    for (size_t k = 0; k < N; ++k) {
      if (used_rows[k] == false) {
        // find column pivot in the remaining rows
        if (firsttime) {
          pivot = matA(k, j);
          pivot_indices[j] = k;
          firsttime = false;
        } else {
          if (abs(matA(k, j)) > abs(pivot)) {
            pivot = matA(k, j);
            pivot_indices[j] = k;
          }
        }
      }
    }
    used_rows[pivot_indices[j]] = true;
    for (size_t k = 0; k < N; ++k) {
      if (used_rows[k] == false) {
        const double factor = matA(k, j) / pivot;
#ifdef DEBUG
        std::cout << "k = " << k << " ; factor = " << factor << std::endl;
#endif
        for (size_t i = j; i < N; ++i) {
          matA(k, i) = matA(k, i) - matA(pivot_indices[j], i) * factor;
        }
        for (size_t i = 0; i < M; ++i) {
          matB(k, i) = matB(k, i) - matB(pivot_indices[j], i) * factor;
        }
      }
    }
#ifdef DEBUG
    std::cout << "Matrix A:\n";
    matA.print(std::cout) << '\n';
#endif
  }
#ifdef DEBUG
  std::cout << "pivot_indices:\n";
  for (const auto& i : pivot_indices) {
    std::cout << i << std::endl;
  }
#endif
  // solve X, backsubstitution
  Matrix matX(N, M);
  // boundary check
  if (N > 0) {
    for (int64_t j = N - 1; j >= 0; --j) {
      // first, we need to find which row has the last pivot
      const size_t l = pivot_indices[j];
      for (size_t i = 0; i < M; ++i) {
        if (j == int64_t(N - 1)) {
          matX(j, i) = matB(l, i) / matA(l, j);
        } else if (j == int64_t(N - 2)) {
          matX(j, i) = (matB(l, i) - matX(j+1, i) * matA(l, j+1)) / matA(l, j);
        } else {
          double sum = 0.0;
          for (size_t k = 1; k < N - j; ++k) {
            sum += matX(j+k, i) * matA(l, j+k);
          }
          matX(j, i) = (matB(pivot_indices[j], i) - sum) / matA(l, j);
        }
      }
    }
  }
  return matX;
}

Matrix CholeskyDecomposition(const Matrix& matA) {
  if (!matA.isSquare()) {
    throw std::invalid_argument("Cholesky decomposition is only implemented for square matrices.");
  }
  Matrix matL(matA.numRows(), matA.numColumns());
  // loop over columns
  for (size_t j = 0; j < matL.numColumns(); ++j) {
    for (size_t i = j; i < matL.numRows(); ++i) {
      double sum = 0.0;
      for (size_t k = 0; k < i; ++k) {
        sum += matL(i, k) * matL(j, k);
      }
      if (i == j) {
        const double l_ii = matA(i, i) - sum;
        if (l_ii < 0) {
          throw std::invalid_argument("The input matrix is not positive-definite.");
        }
        matL(i, i) = std::sqrt(l_ii);
      } else {
        // i > j
        matL(i, j) = 1.0 / matL(j, j) * (matA(j, i) - sum);
      }
    }
  }
  return matL;
}

tuple<Matrix, Matrix> GramSchmidtProcess(const Matrix& matA) {
  if (!matA.isSquare()) {
    throw std::invalid_argument("Gram-Schmidt process is only implemented for square matrices.");
  }
  Matrix Q(matA.numRows(), matA.numColumns());
  Matrix R(matA.numRows(), matA.numColumns());
//   std::vector<double> proj(matA.numRows(), 0);
  for (size_t j = 0; j < matA.numColumns(); ++j) {
    // sum of projection
    std::vector<double> proj_sum(matA.numRows(), 0);
    for (size_t k = 0; k < j; ++k) {
      double numerator = 0;
      double denominator = 0;
      // maybe I need to do column pivoting here?
      for (size_t i = 0; i < matA.numRows(); ++i) {
        numerator += matA(i, j) * Q(i, k);
        denominator += Q(i, k) * Q(i, k);
      }
      R(k, j) = numerator / denominator;
      for (size_t i = 0; i < matA.numRows(); ++i) {
        proj_sum[i] += Q(i, k) * R(k, j);
      }
    }
    if (j == 0) {
      for (size_t i = 0; i < matA.numRows(); ++i) {
        Q(i, j) = matA(i, j);
      }
    } else {
      for (size_t i = 0; i < matA.numRows(); ++i) {
        Q(i, j) = matA(i, j) - proj_sum[i];
      }
    }
    // normalization
    for (size_t i = 0; i < matA.numRows(); ++i) {
      R(j, j) += Q(i, j) * Q(i, j);
    }
    R(j, j) = std::sqrt(R(j, j));
    for (size_t i = 0; i < matA.numRows(); ++i) {
      Q(i, j) = Q(i, j) / R(j, j);
    }
  }
  return std::make_tuple(Q, R);
}
