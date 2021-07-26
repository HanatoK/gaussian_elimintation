#include "Matrix.h"
#include "Spline.h"
#include "Reparametrization.h"
#include "testMatrices.h"

#include <fmt/format.h>
#include <fstream>

void testReparametrization() {
  std::cout << "=======testReparametrization starts=======\n";
  std::ifstream ifs("../test_data/path_input.txt");
  Matrix mat(ifs);
  std::cout << "Input nodes:\n" << mat;
  std::vector<double> distances = calcDistance(mat);
  for (size_t i = 0; i < distances.size(); ++i) {
    fmt::print("Distance between image {:5d} and {:5d}: {:12.7f}\n", i, i+1, distances[i]);
  }
  Reparametrization reparam(mat);
  Matrix result = reparam.compute();
  std::cout << "Reparametrization:\n" << result;
  distances = calcDistance(result);
  for (size_t i = 0; i < distances.size(); ++i) {
    fmt::print("Distance between image {:5d} and {:5d}: {:12.7f}\n", i, i+1, distances[i]);
  }
  std::cout << "=======testReparametrization ends=======\n";
}

int main() {
  testReparametrization();
  return 0;
}
