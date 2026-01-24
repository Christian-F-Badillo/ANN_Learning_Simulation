#include "data_loader.h"
#include <charconv>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dataLoader {

DataLoader::DataLoader(const std::string &fullPath) : path(fullPath) {}

void DataLoader::loadData() {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("File not found");

  std::string line;

  while (std::getline(file, line)) {
    if (line.size() < 2)
      continue;

    std::vector<int> row;
    row.reserve(64);

    std::string_view view(line);

    std::string_view dataPart = view.substr(0, view.size() - 2);

    int label = view.back() - '0';

    size_t start = 0;
    size_t end = dataPart.find(',');

    while (end != std::string_view::npos) {
      std::string_view token = dataPart.substr(start, end - start);

      int value = 0;
      std::from_chars(token.data(), token.data() + token.size(), value);
      row.push_back(value);

      start = end + 1;
      end = dataPart.find(',', start);
    }

    std::string_view lastToken = dataPart.substr(start);
    int lastValue = 0;
    std::from_chars(lastToken.data(), lastToken.data() + lastToken.size(),
                    lastValue);
    row.push_back(lastValue);

    features.push_back(std::move(row));
    labels.push_back(label);
  }
}

const std::vector<std::vector<int>> &DataLoader::getFeatures() {
  return features;
}
const std::vector<int> &DataLoader::getLabels() { return labels; }

} // namespace dataLoader
