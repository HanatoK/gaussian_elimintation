#include "Reparametrization.h"

Reparametrization::Reparametrization
  (const Matrix& matA, size_t resolution_factor):
  m_input(matA), m_num_images(matA.numRows()), m_resolution(m_num_images * resolution_factor) {}

Reparametrization::Reparametrization(const Matrix& matA, size_t num_images, size_t resolution_factor):
  m_input(matA), m_num_images(num_images), m_resolution(m_num_images * resolution_factor) {}

Matrix Reparametrization::interpolate() const {
  Matrix m_interp(m_resolution, m_input.numColumns());
  std::vector<double> t(m_input.numRows());
  std::vector<double> ft(m_input.numRows());
  bool index_ok = true;
  for (size_t i = 0; i < m_input.numRows(); ++i) {
    t[i] = double(i) / (m_input.numRows() - 1);
  }
  for (size_t j = 0; j < m_input.numColumns(); ++j) {
    for (size_t i = 0; i < m_input.numRows(); ++i) {
      ft[i] = m_input(i, j);
    }
    SplineInterpolate spline(t, ft, true);
    for (size_t i = 0; i < m_resolution; ++i) {
      double x = double(i) / (m_resolution - 1);
      m_interp(i, j) = spline.evaluate(x, &index_ok);
      if (index_ok == false) {
        std::cerr << "BUG at Matrix Reparametrization::interpolate\n";
      }
    }
  }
  return m_interp;
}

// calculate the distance between point i and j in matrix
double Reparametrization::distance(const Matrix& matA, size_t i, size_t j) {
  double sum = 0.0;
  for (size_t k = 0; k < matA.numColumns(); ++k) {
    const double tmp = matA(i, k) - matA(j, k);
    sum += tmp * tmp;
  }
  return std::sqrt(sum);
}

Matrix Reparametrization::compute() const {
  Matrix interpolate_matrix = interpolate();
  Matrix result(m_num_images, m_input.numColumns());
  // copy the first and the last image
  for (size_t j = 0; j < m_input.numColumns(); ++j) {
    result(0, j) = m_input(0, j);
    result(result.numRows() - 1, j) = m_input(m_input.numRows() - 1, j);
  }
  std::vector<double> L(m_resolution);
  L[0] = 0;
  for (size_t i = 1; i < m_resolution; ++i) {
    const double dist =
      Reparametrization::distance(interpolate_matrix, i, i-1);
    L[i] = dist + L[i-1];
  }
  const double total_L = L.back();
  size_t k_lower = 0;
  for (size_t i = 1; i < m_num_images - 1; ++i) {
    const double l = total_L * double(i) / (m_num_images - 1);
    // find an index k of array L, where L[k] < l and L[k+1] > l
    // actually L is monotonically increasing so we do not need
    // a full bracketing
    // fmt::print("k = {:6d} ; L[k] = {:10.5f} ; l = {:10.5f}\n", k_lower, L[k_lower], l);
    while (L[k_lower] < l) {
      ++k_lower;
    }
    k_lower = k_lower > 0 ? k_lower - 1 : k_lower;
    for (size_t j = 0; j < m_input.numColumns(); ++j) {
      result(i, j) = interpolate_matrix(k_lower, j);
    }
  }
  // std::cout << "Interpolation matrix:\n" << interpolate_matrix;
  return result;
}
