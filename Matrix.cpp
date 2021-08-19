#include "Matrix.h"

#include <stdexcept>
#include <random>
#include <fmt/format.h>

Matrix::Matrix(): m_nrows(0), m_ncols(0) {}

Matrix::Matrix(std::istream& ifs, const std::string& delimiter) {
  std::string line;
  std::vector<std::string> fields;
  std::vector<std::vector<double>> vec;
  while (std::getline(ifs, line)) {
    fields.clear();
    splitString(line, delimiter, fields);
    if (fields.size() > 0) {
      vec.push_back(std::vector<double>(fields.size()));
      for (size_t i = 0; i < fields.size(); ++i) {
        vec.back()[i] = std::stod(fields[i]);
      }
    }
  }
  fromVector(vec);
}

Matrix::Matrix(const vector<vector<double>>& vec) {
  fromVector(vec);
}

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

void Matrix::fromVector(const vector<vector<double>>& vec) {
  m_nrows = vec.size();
  m_ncols = vec.begin()->size();
  m_data.resize(m_nrows * m_ncols);
  size_t index = 0;
  for (const auto& r: vec) {
    if (r.size() != m_ncols) {
      throw std::invalid_argument("Rows have different numbers of elements.");
    }
    for (const auto& elem: r) {
      m_data[index++] = elem;
    }
  }
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

Matrix Matrix::identity(size_t N) {
  Matrix result(N, N);
  for (size_t i = 0; i < N; ++i) {
    result(i, i) = 1.0;
  }
  return result;
}

Matrix Matrix::rand(size_t N, double a, double b) {
  Matrix result(N, N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(a, b);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      result(i, j) = dis(gen);
    }
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

double Matrix::rootMeanSquareError(const Matrix& matA, const Matrix& matB) {
  if (matA.m_data.size() != matB.m_data.size()) {
    throw std::invalid_argument("RMSE requires the two matrices have the same size.");
  }
  double mse = 0.0;
  for (size_t i = 0; i < matA.m_data.size(); ++i) {
    const double diff = matA.m_data[i] - matB.m_data[i];
    mse += diff * diff;
  }
  return std::sqrt(mse / matA.m_data.size());
}

LUDecomposition::LUDecomposition(Matrix matA, bool LUP): m_LUP(LUP),
  m_matL(matA.numRows(), matA.numRows()),
  m_matU(matA.numRows(), matA.numRows()),
  m_matP(matA.numRows(), matA.numRows()),
  m_inv_matL(matA.numRows(), matA.numRows()),
  m_inv_matU(matA.numRows(), matA.numRows()) {
  if (!matA.isSquare()) {
    throw std::invalid_argument("LUDecomposition is only available for square matrix.");
  }
  const size_t N = matA.numRows();
  for (size_t i = 0; i < N; ++i) {
    m_matL(i, i) = 1.0;
    if (m_LUP) m_matP(i, i) = 1.0;
  }
  for (size_t j = 0; j < N; ++j) {
    if (m_LUP) {
      double Umax = 0.0;
      size_t current_row = j;
      for (size_t r = j; r < N; ++r) {
        const double Uii = matA(r, j);
        if (std::abs(Uii) > Umax) {
          Umax = std::abs(Uii);
          current_row = r;
        }
      }
      if (j != current_row) {
        matA.swapRows(j, current_row);
        m_matP.swapRows(j, current_row);
      }
    }
  }
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < j + 1; ++i) {
      double sum = 0.0;
      for (size_t k = 0; k < i; ++k) {
        sum += m_matL(i, k) * m_matU(k, j);
      }
      m_matU(i, j) = matA(i, j) - sum;
    }
    for (size_t i = j + 1; i < N; ++i) {
      double sum = 0.0;
      for (size_t k = 0; k < j; ++k) {
        sum += m_matL(i, k) * m_matU(k, j);
      }
      m_matL(i, j) = (matA(i, j) - sum) / m_matU(j, j);
    }
  }
  inverse_matL();
  inverse_matU();
}

const Matrix& LUDecomposition::getL() const {
  return m_matL;
}

const Matrix& LUDecomposition::getU() const {
  return m_matU;
}

const Matrix& LUDecomposition::getP() const {
  return m_matP;
}

const Matrix& LUDecomposition::getInverseL() const {
  return m_inv_matL;
}

const Matrix& LUDecomposition::getInverseU() const {
  return m_inv_matU;
}

void LUDecomposition::inverse_matU() {
  // backsubstitution
  const int N = m_matU.numRows();
  const Matrix matI = Matrix::identity(N);
  if (N > 0) {
    for (int j = N - 1; j >= 0; --j) {
      for (int i = 0; i < N; ++i) {
        if (j == (int)(N - 1)) {
          m_inv_matU(j, i) = matI(j, i) / m_matU(j, j);
        } else {
          double sum = 0.0;
          for (int k = 1; k < N - j; ++k) {
            sum += m_inv_matU(j+k, i) * m_matU(j, j+k);
          }
          m_inv_matU(j, i) = (matI(j, i) - sum) / m_matU(j, j);
        }
      }
    }
  }
}

void LUDecomposition::inverse_matL() {
  const size_t N = m_matU.numRows();
  const Matrix matI = Matrix::identity(N);
  if (N > 0) {
    // iterate over columns
    for (size_t j = 0; j < N; ++j) {
      for (size_t i = 0; i < N; ++i) {
        if (i == 0) {
          m_inv_matL(i, j) = matI(i, j) / m_matL(i, i);
        } else {
          double sum = 0.0;
          for (size_t k = 0; k < i; ++k) {
            sum += m_matL(i, k) * m_inv_matL(k, j);
          }
          m_inv_matL(i, j) = (matI(i, j) - sum) / m_matL(i, i);
        }
      }
    }
  }
}

Matrix LUDecomposition::inverse() const {
  const size_t N = m_matU.numRows();
  const Matrix matI = Matrix::identity(N);
  return m_inv_matU * m_inv_matL * m_matP * matI;
}

realSymmetricEigenSolver::realSymmetricEigenSolver(
  const Matrix& matA, double threshold):
  m_matA(matA), m_matV(Matrix::identity(m_matA.numRows())), m_threshold(threshold) {
}

tuple<Matrix, Matrix> realSymmetricEigenSolver::solve() {
  const double num_diagonal_elements = m_matA.numRows() * (m_matA.numRows() - 1) / 2.0;
  double off_diag_sum = std::sqrt(m_matA.diagonalSquaredSum() / num_diagonal_elements);
  // is this enough?
  const size_t max_iteration = 20;
  size_t iteration = 0;
  while (off_diag_sum > m_threshold) {
    JacobiSweep();
    off_diag_sum = std::sqrt(m_matA.diagonalSquaredSum() / num_diagonal_elements);
    ++iteration;
    std::cout << fmt::format("Sweep {:02d}: off-diagonal squared sum = {:8.4f}\n", iteration, off_diag_sum);
    if (iteration >= max_iteration) {
      std::cerr << "Maximum iterations reached in Eigen solver!\n";
      break;
    }
  }
  return std::make_tuple(m_matA, m_matV);
}

void realSymmetricEigenSolver::calc_c_s(
  double a_pq, double a_pp, double a_qq, double& c, double& s) {
  const double theta = 0.5 * (a_qq - a_pp) / a_pq;
  const double sign = sgn(theta) == 0 ? 1.0 : sgn(theta);
  const double t = sign / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
  c = 1.0 / std::sqrt(t * t + 1.0);
  s = t * c;
}

void realSymmetricEigenSolver::applyJacobiTransformation(
  double c, double s, size_t p, size_t q) {
  const double c2 = c*c;
  const double s2 = s*s;
  const double cs = c*s;
  // is it possible to optimize the copy?
  const Matrix old_matrix(m_matA);
  for (size_t i = 0; i < m_matA.numRows(); ++i) {
    for (size_t j = 0; j < m_matA.numColumns(); ++j) {
      if (i != p && i != q) {
        if (j == p) {
          m_matA(i, p) = c * old_matrix(i, p) - s * old_matrix(i, q);
          m_matA(p, i) = m_matA(i, p);
        }
        if (j == q) {
          m_matA(i, q) = c * old_matrix(i, q) + s * old_matrix(i, p);
          m_matA(q, i) = m_matA(i, q);
        }
      }
    }
  }
  m_matA(p, p) = c2 * old_matrix(p, p) + s2 * old_matrix(q, q) -
                  2.0 * cs * old_matrix(p, q);
  m_matA(q, q) = s2 * old_matrix(p, p) + c2 * old_matrix(q, q) +
                  2.0 * cs * old_matrix(p, q);
  m_matA(p, q) = 0;
  m_matA(q, p) = m_matA(p, q);
}

void realSymmetricEigenSolver::multiplyJacobi(
  double c, double s, size_t p, size_t q) {
  // is it possible to optimize the copy?
  const Matrix old_matrix(m_matV);
  for (size_t i = 0; i < m_matV.numRows(); ++i) {
    for (size_t j = 0; j < m_matV.numColumns(); ++j) {
      if (j == p) m_matV(i, p) = c * old_matrix(i, p) - s * old_matrix(i, q);
      if (j == q) m_matV(i, q) = s * old_matrix(i, p) + c * old_matrix(i, q);
    }
  }
}

void realSymmetricEigenSolver::JacobiSweep() {
  const size_t nrows = m_matA.numRows();
  const size_t ncols = m_matA.numColumns();
  double c, s;
  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = i + 1; j < ncols; ++j) {
      const double a_pq = m_matA(i, j);
      const double a_pp = m_matA(i, i);
      const double a_qq = m_matA(j, j);
      if (std::abs(a_pq) > std::numeric_limits<double>::epsilon()) {
        realSymmetricEigenSolver::calc_c_s(a_pq, a_pp, a_qq, c, s);
        applyJacobiTransformation(c, s, i, j);
        multiplyJacobi(c, s, i, j);
      }
    }
  }
}

Matrix getHouseholderPLeft(
  const Matrix& matA, const size_t col, const size_t row) {
  if (row >= matA.numRows()) {
    const std::string err = "Access row " + std::to_string(row) +
       " but matrix has only " + std::to_string(matA.numRows()) +
       " row(s)!\n";
    throw std::out_of_range(err.c_str());
  }
  if (col >= matA.numColumns()) {
    const std::string err = "Access column " + std::to_string(col) +
       " but matrix has only " + std::to_string(matA.numColumns()) +
       " column(s)!\n";
    throw std::out_of_range(err.c_str());
  }
  // get a column vector from A and construct a left householder matrix
  Matrix P = Matrix::identity(matA.numRows());
  std::vector<double> u(matA.numRows() - row, 0);
  double norm2 = 0;
  for (size_t i = row; i < matA.numRows(); ++i) {
    const size_t index_u = i - row;
    u[index_u] = matA(i, col);
    norm2 += u[index_u] * u[index_u];
  }
  double H = norm2 - u[0] * u[0];
  const double norm = std::sqrt(norm2);
  const double sign = sgn(matA(col, row));
  u[0] = u[0] + (sign == 0 ? 1.0 : double(sign)) * norm;
  H = 0.5 * (H + u[0] * u[0]);
  for (size_t i = row; i < matA.numRows(); ++i) {
    const size_t index_u_i = i - row;
    for (size_t j = i; j < matA.numRows(); ++j) {
      const size_t index_u_j = j - row;
      P(i, j) = P(i, j) - u[index_u_i] * u[index_u_j] / H;
      // P.transpose() == P
      if (j > i) {
        P(j, i) = P(i, j);
      }
    }
  }
  return P;
}

Matrix getHouseholderPRight(const Matrix& matA, const size_t col, const size_t row) {
  if (row >= matA.numRows()) {
    const std::string err = "Access row " + std::to_string(row) +
       " but matrix has only " + std::to_string(matA.numRows()) +
       " row(s)!\n";
    throw std::out_of_range(err.c_str());
  }
  if (col >= matA.numColumns()) {
    const std::string err = "Access column " + std::to_string(col) +
       " but matrix has only " + std::to_string(matA.numColumns()) +
       " column(s)!\n";
    throw std::out_of_range(err.c_str());
  }
  Matrix P = Matrix::identity(matA.numColumns());
  std::vector<double> u(matA.numColumns() - col, 0);
  double norm2 = 0;
  for (size_t j = col; j < matA.numColumns(); ++j) {
    const size_t index_u = j - col;
    u[index_u] = matA(row, j);
    norm2 += u[index_u] * u[index_u];
  }
  double H = norm2 - u[0] * u[0];
  const double norm = std::sqrt(norm2);
  const int sign = sgn(matA(col, row));
  u[0] = u[0] + (sign == 0 ? 1.0 : double(sign)) * norm;
  H = 0.5 * (H + u[0] * u[0]);
  for (size_t j = col; j < matA.numColumns(); ++j) {
    const size_t index_u_j = j - col;
    for (size_t i = j; i < matA.numColumns(); ++i) {
      const size_t index_u_i = i - col;
      P(i, j) = P(i, j) - u[index_u_i] * u[index_u_j] / H;
      // P.transpose() == P
      if (j < i) {
        P(j, i) = P(i, j);
      }
    }
  }
  return P;
}

tuple<Matrix, Matrix> HouseholderQRNaive(const Matrix& matA) {
  // naive implementation (need further optimization)
  Matrix R = matA;
  Matrix Q = Matrix::identity(matA.numRows());
  size_t i = 0, j = 0;
  while (i < matA.numRows() && j < matA.numColumns()) {
    const Matrix P = getHouseholderPLeft(
      R, i, j);
    std::cout << "P = \n" << P;
    Q = Q * P;
    R = P * R;
    std::cout << "R = \n" << R;
    std::cout << "Q = \n" << Q;
    ++i;
    ++j;
  }
  return std::make_tuple(Q, R);
}

tuple<Matrix, Matrix> HouseholderQR(const Matrix& matA) {
  const size_t M = matA.numRows();
  const size_t N = matA.numColumns();
  Matrix R = matA;
  Matrix R_previous = R;
  Matrix P(M, M);
  Matrix Q = Matrix::identity(M);
  Matrix Q_previous = Q;
  size_t row = 0, col = 0;
  double norm2 = 0, H = 0, norm = 0;
  std::vector<double> u;
  u.reserve(M);
  while (row < M && col < N) {
    u.clear();
    norm2 = 0;
    for (size_t i = row; i < M; ++i) {
      u.push_back(R(i, col));
      norm2 += R(i, col) * R(i, col);
      for (size_t j = row; j < M; ++j) {
        if (i == j) P(i, j) = 1;
        else P(i, j) = 0;
      }
    }
    H = norm2 - u[0] * u[0];
    norm = std::sqrt(norm2);
    u[0] += (sgn(R(col, row)) == 0 ? 1.0 : sgn(R(col, row))) * norm;
    H = 0.5 * (H + u[0] * u[0]);
    // P and Q are square matrices
    for (size_t i = row; i < M; ++i) {
      const size_t index_u_i = i - row;
      for (size_t j = i; j < M; ++j) {
        const size_t index_u_j = j - row;
        P(i, j) = P(i, j) - u[index_u_i] * u[index_u_j] / H;
        // P.transpose() == P
        if (j > i) {
          P(j, i) = P(i, j);
        }
      }
      // implicitly apply the Householder transformation at the left
      for (size_t j = 0; j < N; ++j) {
        double sum = 0.0;
        // P.numcolumns() == R.numRows()
        for (size_t k = row; k < M; ++k) {
          sum += P(i, k) * R_previous(k, j);
        }
        R(i, j) = sum;
      }
      for (size_t j = 0; j < M; ++j) {
        double sum = 0.0;
        for (size_t k = row; k < M; ++k) {
          sum += Q_previous(j, k) * P(k, i);
        }
        Q(j, i) = sum;
      }
    }
    ++row;
    ++col;
    R_previous = R;
    Q_previous = Q;
  }
  return std::make_tuple(Q, R);
}

tuple<Matrix, Matrix, Matrix> naiveBidiagonalization(Matrix matA) {
  const size_t M = matA.numRows();
  const size_t N = matA.numColumns();
  const size_t K = std::min(M, N);
  Matrix P_left = Matrix::identity(M);
  Matrix P_right = Matrix::identity(N);
  for (size_t k = 0; k < K; ++k) {
    auto tmp = getHouseholderPLeft(matA, k, k);
    P_left = P_left * tmp;
    matA = tmp * matA;
    if (k + 1 < K) {
      tmp = getHouseholderPRight(matA, k + 1, k);
      P_right = tmp * P_right;
      matA = matA * tmp;
    }
  }
  return std::make_tuple(P_left, matA, P_right);
}

tuple<Matrix, Matrix, Matrix> SVDPhaseTwo(const Matrix& matA) {
  const size_t M = matA.numRows();
  const size_t N = matA.numColumns();
  const size_t K = std::min(M, N);
  Matrix matH(K * 2, K * 2);
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = K; j < K * 2; ++j) {
      matH(i, j) = matA(j - K, i);
      matH(j, i) = matH(i, j);
    }
  }
  std::cout << "Matrix H:\n" << matH;
  realSymmetricEigenSolver solver(matH);
  tuple<Matrix, Matrix> eigen = solver.solve();
  const Matrix& Lambda = std::get<0>(eigen);
  const Matrix& Q = std::get<1>(eigen);
  std::cout << "Matrix Q:\n" << Q;
  // const Matrix tmp = Q * Lambda * Q.transpose();
  // std::cout << "Q*Λ*Q'\n" << tmp;
  // std::cout << "RMSE = " << Matrix::rootMeanSquareError(tmp, matH) << std::endl;
  // I get H = Q*Λ*Q' but H*Q = Q*Λ is required
  // FIXME: what should I do? multiplying sqrt(2.0)?
  Matrix matU(M, M);
  Matrix matV(N, N);
  Matrix matSigma(M, N);
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = 0; j < K; ++j) {
      matV(j, i) = std::sqrt(2.0) * Q(i, j);
      matU(i, j) = std::sqrt(2.0) * Q(i + K, j);
      if (i == j) matSigma(i, j) = Lambda(i, j);
    }
  }
  return std::make_tuple(matU, matSigma, matV);
}

Matrix GaussianElimination(Matrix& matA, Matrix& matB) {
  // assume matA is always a square matrix
  const size_t N = matA.numRows();
  if (!matA.isSquare()) {
    throw std::invalid_argument("Gaussian elimination is only implemented for square matrices.");
  }
  // B has M columns
  const size_t M = matB.numColumns();
  // std::cerr << "N = " << N << " ; M = " << M << std::endl;
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
  for (size_t j = 0; j < matA.numColumns(); ++j) {
    for (size_t i = 0; i < matA.numRows(); ++i) {
      Q(i, j) = matA(i, j);
    }
    // sum of projection
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
        Q(i, j) -= Q(i, k) * R(k, j);
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

tuple<Matrix, Matrix> ModifiedGramSchmidtProcess(const Matrix& matA) {
  Matrix Q(matA);
  Matrix R(matA.numColumns(), matA.numColumns());
  for (size_t j = 0; j < Q.numColumns(); ++j) {
    // normalization
    for (size_t i = 0; i < Q.numRows(); ++i) {
      R(j, j) += Q(i, j) * Q(i, j);
    }
    R(j, j) = std::sqrt(R(j, j));
    double denominator = 0;
    for (size_t i = 0; i < Q.numRows(); ++i) {
      Q(i, j) = Q(i, j) / R(j, j);
      denominator += Q(i, j) * Q(i, j);
    }
    // project all following vectors to j, and subtract the projection
    for (size_t k = j + 1; k < Q.numColumns(); ++k) {
      double numerator = 0;
      // calculate the projection
      for (size_t i = 0; i < Q.numRows(); ++i) {
        numerator += Q(i, j) * Q(i, k);
      }
      R(j, k) = numerator / denominator;
      // subtract all following vectors
      for (size_t i = 0; i < Q.numRows(); ++i) {
        Q(i, k) -= Q(i, j) * R(j, k);
      }
    }
  }
  return std::make_tuple(Q, R);
}

vector<double> calcDistance(const Matrix& mat) {
  std::vector<double> distances;
  for (size_t i = 1; i < mat.numRows(); ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < mat.numColumns(); ++j) {
      const double diff = mat(i, j) - mat(i-1, j);
      sum += diff * diff;
    }
    distances.push_back(std::sqrt(sum));
  }
  return distances;
}
