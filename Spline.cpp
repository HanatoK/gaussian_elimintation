#include "Spline.h"

#include <fmt/format.h>

InterpolateBase::InterpolateBase(const std::vector<double>& X,
                                 const std::vector<double>& Y,
                                 const size_t M, const bool equidistant):
  m_X(X), m_Y(Y), m_segment_range(M), m_equidistant(equidistant) {
  m_min_dist = X[1] - X[0];
  for (size_t i = 1; i < X.size() - 1; ++i) {
    const double dist = X[i+1] - X[i];
    if (dist < m_min_dist) {
      m_min_dist = dist;
    }
    // fmt::print("dist[{:d}] = {:8.5f}\n", i, dist);
  }
  const size_t N = std::nearbyint((X.back() - X.front()) / m_min_dist) + 1;
  m_index.resize(N);
  // std::cout << "N = " << N << " ; min_dist = " << m_min_dist << std::endl;
  double tmp_x = X[0];
  for (size_t i = 0; i < N; ++i) {
    m_index[i] = locate(tmp_x, nullptr, 2);
    // fmt::print("{:8.5f} {:5d} {:d}\n", tmp_x, i, m_index[i]);
    tmp_x += m_min_dist;
  }
}

size_t InterpolateBase::index(const double x, bool* index_ok) {
  // if (m_equidistant) return fastIndex(x, index_ok);
  // else return locate(x, index_ok);
  if (x < m_X.front() || x > m_X.back()) {
    // boundary check
    if (index_ok != nullptr) {
      (*index_ok) = false;
    }
    return 0;
  }
  const size_t i = std::floor((x - m_X.front()) / m_min_dist);
  const size_t lower_index = m_index[i];
  int index = lower_index - static_cast<int>((m_segment_range - 2) / 2.0);
  index = std::min(static_cast<int>(m_X.size() - m_segment_range), index);
  if (index > 0) return index;
  else return 0;
}

size_t InterpolateBase::fastIndex(const double x, bool* index_ok) {
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
  const int lower_index = std::floor((x - m_X[0]) / step);
  int index = lower_index - static_cast<int>((m_segment_range - 2) / 2.0);
  index = std::min(static_cast<int>(m_X.size() - m_segment_range), index);
  if (index > 0) return index;
  else return 0;
}

size_t InterpolateBase::locate(const double x, bool* index_ok,
                               long long int M) {
  // given a value x, return an index such that x is centered in the
  // subrange from X[j] to X[j+M]
  // assume m_X is sorted to be monotonically increasing
  if (x < m_X.front() || x > m_X.back()) {
    // boundary check
    if (index_ok != nullptr) {
      (*index_ok) = false;
    }
    return 0;
  }
  if (M < 0) M = m_segment_range;
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
  // std::cout << "lower_index = " << lower_index << std::endl;
  int index = lower_index - static_cast<int>((M - 2) / 2.0);
  index = std::min(static_cast<int>(m_X.size() - M), index);
  if (index > 0) return index;
  else return 0;
}

double InterpolateBase::rawInterp(const size_t table_index, const double x) {
  std::cerr << "rawInterp is not implemented.\n";
  return 0;
}