#include "Matrix.h"
#include "Spline.h"
#include "Reparametrization.h"

#include <fmt/format.h>
#include <fstream>
#include <boost/program_options.hpp>

bool reparametrization(
  const std::string& input_filename, const std::string& output_filename, const int num_images) {
  std::ifstream ifs(input_filename.c_str());
  if (!ifs.is_open()) {
    std::cerr << "Cannot open input file " + input_filename + "\n";
    return false;
  }
  const Matrix mat(ifs);
  const std::vector<double> old_distances = calcDistance(mat);
  const Reparametrization reparam = num_images <= 0 ? Reparametrization(mat) : Reparametrization(mat, size_t(num_images), 1000);
  const Matrix result = reparam.compute();
  const std::vector<double> new_distances = calcDistance(result);
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
    ("num_images,n", boost::program_options::value<int>()->default_value(0), "The number of images. 0 or negative values are default to the number of input images");
  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);
  if (vm.count("help")) {
    std::cout << desc << '\n';
  } else {
    const std::string input_filename = vm["input"].as<std::string>();
    const std::string output_filename = vm["output"].as<std::string>();
    const int num_images = vm["num_images"].as<int>();
    if (!reparametrization(input_filename, output_filename, num_images)) return 1;
  }
  return 0;
}
