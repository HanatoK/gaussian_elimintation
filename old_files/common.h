#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>

void splitString(const std::string& data, const std::string& delim, std::vector<double>& dest);
std::vector<std::vector<double>> readTo2DVector(const std::string& filename);
void PrintMatrix(const std::vector<std::vector<double>>& mat, std::ostream& os = std::cout);

#endif // COMMON_H
