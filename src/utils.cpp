#include "utils.hpp"
#include <algorithm>

void trimLeft(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) {
    return !std::isspace(c);
  }));
}

void trimRight(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c) {
    return !std::isspace(c);
  }).base(), s.end());
}
