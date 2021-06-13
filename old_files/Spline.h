#ifndef SPLINE_H
#define SPLINE_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "Matrix.h"

using std::vector;

class CubicSpline {
public:
  enum class boundary_condition {natural, not_a_knot};
  CubicSpline(const vector<double>& X, const vector<double>& Y,
              boundary_condition bc = boundary_condition::natural);
  double getValue(double x) const;
  double getDerivative(double x) const;
  const vector<double>& getFactorA() const;
  const vector<double>& getFactorB() const;
  const vector<double>& getFactorC() const;
  const vector<double>& getFactorD() const;
private:
  void interpolate();
  size_t m_numPoints;
  boundary_condition m_bc;
  vector<double> m_X;
  vector<double> m_Y;
  vector<double> m_dX;
  vector<double> m_dY;
  vector<double> m_A;
  vector<double> m_B;
  vector<double> m_C;
  vector<double> m_D;
};

#endif // SPLINE_H
