#pragma once

#include <string>
#include <map>
#include <memory>
#include <vector>
#include "math.hpp"

struct BufferItem {
  MathObjectType type;
  size_t offset;
};

struct Buffer {
  std::vector<uint8_t> storage;
  std::map<std::string, BufferItem> items;

  void insert(const std::string& name, Array& item);
  void insert(const std::string& name, Array2& item);
  void insert(const std::string& name, Array3& item);
};

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
    virtual void execute(Buffer& buffer, const Computation& computation) const = 0;

    virtual ~Executor() {}
};

using ExecutorPtr = std::unique_ptr<Executor>;

