#pragma once
#include <string>
#include <vector>

namespace dataLoader {

class DataLoader {

public:
  DataLoader(const std::string &fullPath);

  void loadData();
  const std::vector<std::vector<int>> &getFeatures();
  const std::vector<int> &getLabels();

private:
  std::vector<std::vector<int>> features;
  std::vector<int> labels;
  std::string path;
};
} // namespace dataLoader
