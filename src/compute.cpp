#include "compute.hpp"
#include "math.hpp"

void Buffer::insert(const std::string& name, Array& item) {
  size_t offset = storage.size();
  storage.resize(offset + item.serializedSize());
  item.serialize(storage.data() + offset);
  items.insert({ name, BufferItem{ MathObjectType::Array, offset } });
}

void Buffer::insert(const std::string& name, Array2& item) {
  size_t offset = storage.size();
  storage.resize(offset + item.serializedSize());
  item.serialize(storage.data() + offset);
  items.insert({ name, BufferItem{ MathObjectType::Array2, offset } });
}

void Buffer::insert(const std::string& name, Array3& item) {
  size_t offset = storage.size();
  storage.resize(offset + item.serializedSize());
  item.serialize(storage.data() + offset);
  items.insert({ name, BufferItem{ MathObjectType::Array3, offset } });
}

void ComputationDesc::chain(const ComputationDesc& c) {
  steps.insert(steps.end(), c.steps.begin(), c.steps.end());
}

Computation::~Computation() {}

