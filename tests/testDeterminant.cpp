#include "Matrix.h"
#include "Spline.h"
#include "Reparametrization.h"
#include "testMatrices.h"

#include <fmt/format.h>
#include <fstream>

void testDeterminant(const Matrix& matA) {
  std::cout << "Matrix A:\n" << matA;
  const double det = matA.determinant();
  std::cout << "Determinant of matA: " << det << '\n';
}

int main() {
  testDeterminant(TEST::matA);;
  return 0;
}
