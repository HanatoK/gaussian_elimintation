#include "common.h"

std::vector<std::vector<double>> readTo2DVector(const std::string& filename) {
  std::ifstream ifs(filename);
  std::string line;
  std::vector<std::vector<double>> mat;
  while (std::getline(ifs, line)) {
    std::vector<double> fields;
    splitString(line, " ", fields);
    mat.push_back(std::move(fields));
  }
  return mat;
} 

void splitString(const std::string& data, const std::string& delim, std::vector<double>& dest) {
  if (delim.empty()) {
    dest.push_back(std::stod(data));
    return;
  }
  size_t index = 0, new_index = 0;
  std::string tmpstr;
  while (index != data.length()) {
    new_index = data.find(delim, index);
    if (new_index != std::string::npos) tmpstr = data.substr(index, new_index - index);
    else tmpstr = data.substr(index, data.length());
    if (!tmpstr.empty()) {
      dest.push_back(std::stod(tmpstr));
    }
    if (new_index == std::string::npos) break;
    index = new_index + 1;
  }
}

void PrintMatrix(const std::vector<std::vector<double>>& mat, std::ostream& os) {
  std::ios_base::fmtflags f(os.flags());
  os << std::fixed;
  os << std::setprecision(7);
  for (const auto& v: mat) {
    for (const auto& item : v) {
      os << std::setw(12) << item << " ";
    }
    os << '\n';
  }
  os.flags(f);
}
