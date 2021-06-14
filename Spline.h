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
  size_t index(const double x, bool* index_ok = nullptr) const;
  size_t fastIndex(const double x, bool* index_ok = nullptr) const;
  size_t locate(const double x, bool* index_ok = nullptr) const; // aka as "locate" in NR
protected:
  std::vector<double> m_X;
  std::vector<double> m_Y;
  size_t m_segment_range;
  bool m_equidistant;
};

class SplineInterpolate: public InterpolateBase {
public:
  enum class boundary_condition {natural, not_a_knot};
  SplineInterpolate(const std::vector<double>& X, const std::vector<double>& Y,
                    const bool equidistant = false,
                    boundary_condition bc = boundary_condition::natural);
  virtual ~SplineInterpolate() {}
  double evaluate(const double x, bool* index_ok = nullptr) const;
private:
  void calcFactors();
  boundary_condition m_bc;
  std::vector<double> m_A;
  std::vector<double> m_B;
  std::vector<double> m_C;
  std::vector<double> m_D;
};

#endif // SPLINE_H
