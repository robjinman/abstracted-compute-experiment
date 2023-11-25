#pragma once

#include "math.hpp"
#include <string>
#include <memory>
#include <vector>

class Buffer {
  public:
    virtual void insert(const std::string& name, Array& item) = 0;
    virtual void insert(const std::string& name, Array2& item) = 0;
    virtual void insert(const std::string& name, Array3& item) = 0;

    virtual ~Buffer() {}
};

using BufferPtr = std::unique_ptr<Buffer>;

struct ComputationDesc {
  std::vector<std::string> steps;

  void chain(const ComputationDesc& c);
};

class Computation {
  public:
    virtual ~Computation() = 0;
};

using ComputationPtr = std::unique_ptr<Computation>;

class Executor {
  public:
    virtual ComputationPtr compile(const Buffer& buffer, const ComputationDesc& desc) const = 0;
    virtual void execute(Buffer& buffer, const Computation& computation,
      size_t iterations = 1) const = 0;

    virtual ~Executor() {}
};

using ExecutorPtr = std::unique_ptr<Executor>;

std::vector<std::string> tokenizeCommand(const std::string& command);
