#include "compute.hpp"
#include "utils.hpp"
#include "exception.hpp"

void ComputationDesc::chain(const ComputationDesc& c) {
  steps.insert(steps.end(), c.steps.begin(), c.steps.end());
}

Computation::~Computation() {}

std::vector<std::string> tokenizeCommand(const std::string& command) {
  std::stringstream ss(command);
  std::vector<std::string> tokens;

  std::string token;
  std::getline(ss, token, '=');

  trimLeft(token);
  trimRight(token);

  ASSERT_MSG(token.length() > 0, STR("Syntax error: " << command));
  tokens.push_back(token);

  while (std::getline(ss, token, ' ')) {
    trimLeft(token);
    trimRight(token);

    if (token.length() > 0) {
      tokens.push_back(token);
    }
  }

  return tokens;
}
