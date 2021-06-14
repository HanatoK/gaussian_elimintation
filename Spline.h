#ifndef SPLINE_H
#define SPLINE_H

#include "Matrix.h"

// I don't like NR's idea about "hunt". Since in my cases most of the time X is
// equidistant, I just optimize for them.
class InterpolateBase {
public:
  InterpolateBase(const std::vector<double>& X, const std::vector<double>& Y,
                  const size_t M, const bool equidistant = false);
  virtual ~InterpolateBase() {}
  size_t index(const double x, bool* index_ok = nullptr);
  size_t fastIndex(const double x, bool* index_ok = nullptr);
  size_t locate(const double x, bool* index_ok = nullptr); // aka as "locate" in NR
  virtual double rawInterp(const size_t table_index, const double x);
private:
  std::vector<double> m_X;
  std::vector<double> m_Y;
  size_t m_segment_range;
  bool m_equidistant;
};

#endif // SPLINE_H
