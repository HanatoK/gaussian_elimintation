#include "Spline.h"

InterpolateBase::InterpolateBase(const std::vector<double>& X,
                                 const std::vector<double>& Y,
                                 const size_t M, const bool equidistant):
  m_X(X), m_Y(Y), m_segment_range(M), m_equidistant(equidistant) {}

size_t InterpolateBase::index(const double x, bool* index_ok) {
  if (m_equidistant) return fastIndex(x, index_ok);
  else return locate(x, index_ok);
}

size_t InterpolateBase::fastIndex(const double x, bool* index_ok) {
  // assume the steps are the same
  const double step = m_X[1] - m_X[0];
  // assume m_X is sorted to be monotonically increasing
  const size_t lower_index = std::floor((x - m_X[0]) / step);
  size_t index = lower_index - static_cast<int>((m_segment_range - 2) / 2.0);
  index = std::min(m_X.size() - m_segment_range, index);
  if (index > 0) return index;
  else return 0;
}

size_t InterpolateBase::locate(const double x, bool* index_ok) {
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
  size_t index = lower_index - static_cast<int>((m_segment_range - 2) / 2.0);
  index = std::min(m_X.size() - m_segment_range, index);
  if (index > 0) return index;
  else return 0;
}

double InterpolateBase::rawInterp(const size_t table_index, const double x) {
  std::cerr << "rawInterp is not implemented.\n";
  return 0;
}