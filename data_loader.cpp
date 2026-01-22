#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class DataLoader {
public:
  DataLoader(std::string fullPath) { path = fullPath; }

  void loadData() {
    std::string line, featuresData, labelsData;
    std::ifstream file(path);
    if (file.is_open()) {

      while (std::getline(file, line)) {

        if (!line.size())
          continue;

        featuresData = line.substr(0, line.length() - 2);
        labelsData = line.back();

        features.push_back(featuresData);
        labels.push_back(labelsData);
      }

      file.close();
    }

    else {
      throw std::runtime_error("File not found");
    }
  }

private:
  std::vector<std::string> features{};
  std::vector<std::string> labels{};
  std::string path;
};

int main() {

  std::ifstream file("data/optdigits.tra");

  return 0;
}
