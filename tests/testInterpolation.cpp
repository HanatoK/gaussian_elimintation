#include "Matrix.h"
#include "Spline.h"
#include "Reparametrization.h"
#include "testMatrices.h"

#include <fmt/format.h>
#include <fstream>

void testInterpolateBase() {
  std::cout << "Test InterpolateBase::locate\n";
  std::vector<double> X{
    -2.0, 1.0, 2.0, 4.0, 10.0, 15.0,
    20.5, 66.3, 70.4, 127.0, 211.6, 215.2,
    266.0, 268.0, 300.6, 466,3, 512.0};
  std::vector<double> Y;
  for (const auto& a : X) {
    Y.push_back(std::sin(a));
  }
  InterpolateBase interp(X, Y, 5);
  bool index_ok = true;
  size_t i = interp.locate(80.0, &index_ok);
  std::cout << std::boolalpha << "Index = " << i
            << " (" << index_ok << ")\n";
  std::cout << "Test InterpolateBase::fastIndex\n";
  for (size_t i = 0; i < X.size(); ++i) {
    X[i] = i * 5.0;
    Y[i] = std::sin(X[i]);
  }
  interp = InterpolateBase(X, Y, 5, true);
  i = interp.fastIndex(50.0, &index_ok);
  std::cout << std::boolalpha << "Index = " << i
            << " (" << index_ok << ")\n";
}

void testSplineInterpolation() {
  const size_t N = 10;
  std::vector<double> X(N);
  std::vector<double> Y(N);
  for (size_t i = 0; i < N; ++i) {
    X[i] = i;
    Y[i] = std::sin(double(i) / 2 * M_PI);
    fmt::print("x = {:6f} ; y = {:6f}\n", X[i], Y[i]);
  }
  SplineInterpolate spline_interp_natural(X, Y, true);
  SplineInterpolate spline_interp_not_a_knot(X, Y, true, SplineInterpolate::boundary_condition::not_a_knot);
  for (size_t i = 0; i < N - 1; ++i) {
    const size_t M = (i == N - 2) ? 21 : 20;
    for (size_t j = 0; j < M; ++j) {
      const double tmp_x = double(j) / 20.0 + i;
      const double val = spline_interp_natural.evaluate(tmp_x);
      const double val2 = spline_interp_not_a_knot.evaluate(tmp_x);
      fmt::print("x = {:6f} ; natural = {:12.7f} ; not_a_knot = {:12.7f}\n", tmp_x, val, val2);
    }
  }
}

int main() {
  testInterpolateBase();
  testSplineInterpolation();
  return 0;
}
