#ifndef REPARAMETRIZATION_H
#define REPARAMETRIZATION_H

#include "Spline.h"
#include <fmt/format.h>
#include <cstddef>

class Reparametrization {
public:
  Reparametrization(const Matrix& matA, size_t resolution_factor = 1000);
  Reparametrization(const Matrix& matA, size_t num_images, size_t resolution_factor = 1000);
  Matrix compute() const;
private:
  static double distance(const Matrix& matA, size_t i, size_t j);
  Matrix interpolate() const;
  Matrix m_input;
  size_t m_num_images;
  size_t m_resolution;
};

#endif // REPARAMETRIZATION_H
