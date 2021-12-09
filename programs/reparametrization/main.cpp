#include "Matrix.h"
#include "Spline.h"
#include "Reparametrization.h"

#include <fmt/format.h>
#include <fstream>
#include <boost/program_options.hpp>

bool reparametrization(
  const std::string& input_filename, const std::string& output_filename,
  const int num_images, const int num_iterations = 1) {
  std::ifstream ifs(input_filename.c_str());
  if (!ifs.is_open()) {
    std::cerr << "Cannot open input file " + input_filename + "\n";
    return false;
  }
  if (num_iterations <= 1) {
    std::cerr << "Invalid number of iterations (" << num_iterations << ")\n";
    return false;
  }
  const Matrix mat(ifs);
  Matrix result(mat);
  for (int i = 0; i < num_iterations; ++i) {
    const std::vector<double> old_distances = calcDistance(result);
    const Reparametrization reparam = (num_images <= 0 || i > 0) ?
                                      Reparametrization(result) :
                                      Reparametrization(result, size_t(num_images), 1000);
    result = reparam.compute();
    const std::vector<double> new_distances = calcDistance(result);
    fmt::print("========== Iteration {:d} start:\n", i);
    if (num_images <= 0) {
      for (size_t i = 0; i < mat.numRows(); ++i) {
        double d = 0;
        for (size_t j = 0; j < mat.numColumns(); ++j) {
          const double x = mat(i, j) - result(i, j);
          d += x * x;
        }
        d = std::sqrt(d);
        fmt::print("Drift of image {:5d}: {:12.7f}\n", i, d);
      }
      for (size_t i = 0; i < new_distances.size(); ++i) {
        fmt::print("Distance between image {:5d} and {:5d}: {:12.7f} (origin) {:12.7f} (reparam)\n",
                  i, i+1, old_distances[i], new_distances[i]);
      }
    } else {
      fmt::print("Reparametrization with interpolation...\n");
      fmt::print("Old distances:\n");
      for (size_t i = 0; i < old_distances.size(); ++i) {
        fmt::print("Distance between image {:5d} and {:5d}: {:12.7f} (origin)\n", i, i+1, old_distances[i]);
      }
      fmt::print("After reparametrization:\n");
      for (size_t i = 0; i < new_distances.size(); ++i) {
        fmt::print("Distance between image {:5d} and {:5d}: {:12.7f} (reparam)\n", i, i+1, new_distances[i]);
      }
    }
    fmt::print("========== Iteration {:d} end.\n", i);
  }
  std::ofstream ofs(output_filename.c_str());
  if (!ofs.is_open()) {
    std::cerr << "Cannot open output file " + output_filename + "\n";
    return false;
  }
  // ofs << result;
  for (size_t i = 0; i < result.numRows(); ++i) {
    for (size_t j = 0; j < result.numColumns(); ++j) {
      ofs << fmt::format(" {:15.7f}", result(i, j));
    }
    ofs << std::endl;
  }
  ofs.close();
  return true;
}

int main(int argc, char* argv[]) {
  boost::program_options::options_description desc{"Options"};
  desc.add_options()
    ("help,h", "Help")
    ("input,i", boost::program_options::value<std::string>()->default_value("path_input.txt"), "Input file")
    ("output,o", boost::program_options::value<std::string>()->default_value("path_output.txt"), "Output file")
    ("num_images,n", boost::program_options::value<int>()->default_value(0), "The number of images. 0 or negative values are default to the number of input images")
    ("num_iterations", boost::program_options::value<int>()->default_value(1), "The number of iterations.");
  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);
  if (vm.count("help")) {
    std::cout << desc << '\n';
  } else {
    const std::string input_filename = vm["input"].as<std::string>();
    const std::string output_filename = vm["output"].as<std::string>();
    const int num_images = vm["num_images"].as<int>();
    const int num_iterations = vm["num_iterations"].as<int>();
    if (!reparametrization(input_filename, output_filename, num_images, num_iterations)) return 1;
  }
  return 0;
}
