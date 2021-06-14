#include "Spline.h"

InterpolateBase::InterpolateBase(const std::vector<double>& X,
                                 const std::vector<double>& Y,
                                 const size_t M, const bool equidistant):
  m_X(X), m_Y(Y), m_segment_range(M), m_equidistant(equidistant) {}

size_t InterpolateBase::index(const double x, bool* index_ok) const {
  if (m_equidistant) return fastIndex(x, index_ok);
  else return locate(x, index_ok);
}

size_t InterpolateBase::fastIndex(const double x, bool* index_ok) const {
  if (x < m_X.front() || x > m_X.back()) {
    // boundary check
    if (index_ok != nullptr) {
      (*index_ok) = false;
    }
    return 0;
  }
  // assume the steps are the same
  const double step = m_X[1] - m_X[0];
  // assume m_X is sorted to be monotonically increasing
  const size_t lower_index = std::floor((x - m_X[0]) / step);
  int index = lower_index - static_cast<int>((m_segment_range - 2) / 2.0);
  index = std::min(static_cast<int>(m_X.size() - m_segment_range), index);
  if (index > 0) return index;
  else return 0;
}

size_t InterpolateBase::locate(const double x, bool* index_ok) const {
  // given a value x, return an index such that x is centered in the
  // subrange from X[j] to X[j+m_segment_range]
  // assume m_X is sorted to be monotonically increasing
  if (x < m_X.front() || x > m_X.back()) {
    // boundary check
    if (index_ok != nullptr) {
      (*index_ok) = false;
    }
    return 0;
  }
  int lower_index = 0;
  int upper_index = m_X.size() - 1;
  int middle_index = 0;
  // find the range where x locates
  while (upper_index - lower_index > 1) {
    middle_index = static_cast<int>((lower_index + upper_index) / 2.0);
    if (x >= m_X[middle_index]) {
      lower_index = middle_index;
    } else {
      upper_index = middle_index;
    }
  }
  int index = lower_index - static_cast<int>((m_segment_range - 2) / 2.0);
  index = std::min(static_cast<int>(m_X.size() - m_segment_range), index);
  if (index > 0) return index;
  else return 0;
}

SplineInterpolate::SplineInterpolate(
  const std::vector<double>& X, const std::vector<double>& Y,
  const bool equidistant, boundary_condition bc):
  InterpolateBase(X, Y, 2, equidistant), m_bc(bc) {
  calcFactors();
}

void SplineInterpolate::calcFactors() {
  // solve the equations:
  // -\frac{1}{\Delta x_i}B_i+
  // \frac{2}{\Delta x_{i+1}}B_{i+1}+
  // \frac{1}{\Delta x_{i+1}}B_{i+2}
  // =3\left(\frac{\Delta y_{i+1}}{\Delta x_{i+1}^2}-\frac{\Delta y_{i}}{\Delta x_{i}^2}\right)
  const size_t num_points = m_X.size();
  const size_t N = num_points - 1;
  using std::vector;
  vector<double> dY_dX2(N);
  vector<double> dX(N);
  m_A.resize(N);
  m_B.resize(N);
  m_C.resize(N);
  m_D.resize(N);
  for (size_t i = 0; i < N; ++i) {
    dX[i] = m_X[i+1] - m_X[i];
    dY_dX2[i] = m_Y[i+1] - m_Y[i];
    dY_dX2[i] /= dX[i] * dX[i];
  }
  // there are N unknowns but N - 2 equations ??
  // build the matrix and the vector first
  Matrix lhs_matrix(N, N);
  Matrix rhs_vector(N, 1);
  for (size_t i = 1; i < N; ++i) {
    rhs_vector(i, 1) = 3.0 * (dY_dX2[i] - dY_dX2[i-1]); // ?
    lhs_matrix(i, i-1) = -1.0 / dX[i-1];
    lhs_matrix(i, i) = 2.0 / dX[i];
    if (i < N - 1) {
      lhs_matrix(i, i+1) = 1.0 / dX[i];
    }
  }
  if (m_bc == boundary_condition::natural) {
    // first node
    rhs_vector(0, 1) = 3.0 * dY_dX2[0];
    lhs_matrix(0, 0) = 2.0 / dX[0];
    lhs_matrix(0, 1) = 1.0 / dX[0];
  }
  const Matrix tmp_B = GaussianElimination(lhs_matrix, rhs_vector);
  std::cout << "N = " << N << std::endl;
  for (size_t i = 0; i < N - 1; ++i) {
    m_B[i] = tmp_B(i, 0);
    const double tmp_sum_b = 2.0 * tmp_B(i, 0) + tmp_B(i+1, 0);
    m_C[i] = 3.0 * dY_dX2[i] - tmp_sum_b / dX[i];
  }
  if (m_bc == boundary_condition::natural) {
    m_B[N-1] = tmp_B(N-1, 0);
    m_C[N-1] = 3.0 * (dY_dX2[N-1] - m_B[N-1] / dX[N-1]) / 2.0;
  }
  for (size_t i = 0; i < N - 1; ++i) {
    m_D[i] = (m_C[i+1] - m_C[i]) / (3.0 * m_X[i]);
  }
  if (m_bc == boundary_condition::natural) {
    m_D[N-1] = -m_C[N-1] / dX[N-1];
  }
}

double SplineInterpolate::evaluate(const double x, bool* index_ok) const {
  const size_t ix = index(x, index_ok);
  const double dx = x - m_X[ix];
  const double dx2 = dx * dx;
  const double dx3 = dx2 * dx;
  const double interp_y = m_A[ix] + m_B[ix] * dx + m_C[ix] * dx2 + m_D[ix] * dx3;
  return interp_y;
}