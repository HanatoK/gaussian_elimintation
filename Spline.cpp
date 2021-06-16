#include "Spline.h"

#include <fmt/format.h>

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
  // \Delta X_i C_i + 2(\Delta X_{i+1} + \Delta X_{i})C_{i+1}+\Delta X_{i+1} C_{i+2} = 3(\Delta Y_{i+1} - \Delta Y_{i})
  const size_t num_points = m_X.size();
  const size_t N = num_points - 1;
  using std::vector;
  vector<double> dY(N);
  vector<double> dX(N);
  m_A.resize(N);
  m_B.resize(N);
  m_C.resize(N);
  m_D.resize(N);
  for (size_t i = 0; i < N; ++i) {
    dX[i] = m_X[i+1] - m_X[i];
    dY[i] = m_Y[i+1] - m_Y[i];
  }
  Matrix tmp_mat(N+1, N+1);
  Matrix tmp_vec(N+1, 1);
  for (size_t i = 1; i < N; ++i) {
    tmp_mat(i, i-1) = dX[i-1];
    tmp_mat(i, i) = 2.0 * (dX[i-1] + dX[i]);
    tmp_mat(i, i+1) = dX[i];
    tmp_vec(i, 0) = 3.0 * (dY[i] / dX[i] - dY[i-1] / dX[i-1]);
  }
  if (m_bc == boundary_condition::natural) {
    // std::cout << "Using natural boundary";
    // natural boundary:
    // S_{0}'' (X_0) = 0
    tmp_mat(0, 0) = 1.0;
    tmp_vec(0, 0) = 0.0;
    // although we have only N splines, assuming there is an extra spline,
    // then S_{N}'' (X_N) = 0
    tmp_mat(N, N) = 1.0;
    tmp_vec(N, 0) = 0.0;
  } else if (m_bc == boundary_condition::not_a_knot) {
    // std::cout << "Using not-a-knot boundary";
    // S_{0}''' (X_1) = S_{1}''' (X_1)
    tmp_mat(0, 0) = -dX[1];
    tmp_mat(0, 1) = dX[0] + dX[1];
    tmp_mat(0, 2) = -dX[0];
    tmp_vec(0, 0) = 0.0;
    tmp_mat(N, N-2) = -dX[N-2];
    tmp_mat(N, N-1) = dX[N-2] + dX[N-1];
    tmp_mat(N, N) = -dX[N-1];
    tmp_vec(N, N) = 0.0;
  }
  // std::cout << "Interpolation matrix:\n" << tmp_mat;
  // std::cout << "Interpolation vector:\n" << tmp_vec;
  const Matrix tmp_C = GaussianElimination(tmp_mat, tmp_vec);
  for (size_t i = 0; i < N; ++i) {
    m_C[i] = tmp_C(i, 0);
    m_D[i] = (tmp_C(i+1, 0) - tmp_C(i, 0)) / (3.0 * dX[i]);
    m_B[i] = dY[i] / dX[i] - dX[i] * (2.0 * tmp_C(i, 0) + tmp_C(i+1, 0)) / 3.0;
    m_A[i] = m_Y[i];
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
