#include "Spline.h"

CubicSpline::CubicSpline(
  const vector<double>& X, const vector<double>& Y,
  boundary_condition bc = boundary_condition::natural):
  m_numPoints(X.size()), m_bc(bc), m_X(X), m_Y(Y) {
  m_dX.assign(m_numPoints - 1);
  m_dY.assign(m_numPoints - 1);
  for (size_t i = 0; i < m_numPoints - 1; ++i) {
    m_dX[i] = m_X[i+1] - m_X[i];
    m_dY[i] = m_Y[i+1] - m_Y[i];
  }
  interpolate();
}

void CubicSpline::interpolate() {
  const size_t n = m_numPoints - 1;
  for (size_t i = 0; i < n; ++i) {
    m_A[i] = m_Y[i];
  }
  Matrix spline_matrix(n+1, n+1);
  Matrix spline_vector(n+1, 1);
  if (m_bc == boundary_condition::natural) {
    spline_matrix(0, 0) = 1.0;
    spline_matrix(n, n) = 1.0;
  } else if (m_bc == boundary_condition::not_a_knot) {
    spline_matrix(0, 0) = -m_dX[1];
    spline_matrix(0, 1) = m_dX[0] + m_dX[1];
    spline_matrix(0, 2) = -m_dX[0];
    spline_matrix(n, n) = -m_dX[n-2];
    spline_matrix(n, n-1) = m_dX[n-2] + m_dX[n-1];
    spline_matrix(n, n-2) = -m_dX[n-1];
  }
  for (size_t i = 1; i < n; ++i) {
    spline_matrix(i, i-1) = m_dX[i-1];
    spline_matrix(i, i) = (m_dX[i-1] + m_dX[i]) * 2.0;
    spline_matrix(i, i+1) = m_dX[i];
    spline_vector(i, 0) = (m_dY[i] / m_dX[i] - m_dY[i-1] / m_dX[i-1]) * 6.0;
  }
  spline_vector(0, 0) = 0;
  spline_vector(n, 0) = 0;
  const Matrix result = GaussianElimination(spline_matrix, spline_vector);
  for (size_t i = 0; i < n; ++i) {
    m_C[i] = result(i, 0) / 2.0;
    m_B[i] = m_dY[i] / m_dX[i] - 0.5 * m_dX[i] * result(i, 0) -
             1.0 / 6.0 * (result(i+1, 0) - result(i, 0)) * m_dX[i];
    m_D[i] = (result(i+1, 0) - result(i, 0)) / (6.0 * m_dX[i]);
  }
}
