#include "gpu_compute.hpp"
#include "types.hpp"
#include <map>
#include <cstring>

namespace {

struct GpuBufferItem {
  MathObjectType type;
  Triple shape;
  size_t offset;
};

class GpuBuffer : public Buffer {
  std::vector<double> storage;
  std::map<std::string, GpuBufferItem> items;

  public:
    void insert(const std::string& name, Array& item) override;
    void insert(const std::string& name, Array2& item) override;
    void insert(const std::string& name, Array3& item) override;

  private:
    template<class T>
    void insertItem(const std::string& name, T& item);
};

template<class T>
void GpuBuffer::insertItem(const std::string& name, T& item) {
  size_t size = item.storage().size();
  size_t offset = storage.size();
  storage.resize(offset + size);
  memcpy(storage.data() + offset, item.storage().data(), size * sizeof(double));
  item.setDataPtr(storage.data() + offset);
  items.insert({ name, GpuBufferItem{ item.type(), item.shape(), offset } });
}

void GpuBuffer::insert(const std::string& name, Array& item) {
  insertItem(name, item);
}

void GpuBuffer::insert(const std::string& name, Array2& item) {
  insertItem(name, item);
}

void GpuBuffer::insert(const std::string& name, Array3& item) {
  insertItem(name, item);
}

struct GpuComputationStep {
  size_t shader;
  size_t numWorkgroups;
};

class GpuComputation : public Computation {
  public:
    std::vector<GpuComputationStep> steps;
};

using GpuComputationPtr = std::unique_ptr<GpuComputation>;

GpuComputationStep compileMultiplyCommand(const GpuBuffer& buffer,
  const std::vector<std::string>& tokens) {

  // TODO
}

GpuComputationStep compileAddCommand(const GpuBuffer& buffer,
  const std::vector<std::string>& tokens) {

  // TODO
}

GpuComputationStep compileCommand(const Buffer& buf, const std::string& command) {
  const auto& buffer = dynamic_cast<const GpuBuffer&>(buf);

  std::vector<std::string> tokens = tokenizeCommand(command);

  ASSERT(tokens.size() >= 2);
  const std::string& functionName = tokens[1];

  if (functionName == "multiply") {
    return compileMultiplyCommand(buffer, tokens);
  }
  else if (functionName == "add") {
    return compileAddCommand(buffer, tokens);
  }
  else {
    EXCEPTION("Function '" << functionName << "' not recognised");
  }
}

class GpuExecutor : public Executor {
  public:
    GpuExecutor(Logger& logger);
  
    ComputationPtr compile(const Buffer& buffer, const ComputationDesc& desc) const override;
    void execute(Buffer& buffer, const Computation& computation) const override;

  private:
    Logger& m_logger;
};

GpuExecutor::GpuExecutor(Logger& logger)
  : m_logger(logger) {}

ComputationPtr GpuExecutor::compile(const Buffer& buffer, const ComputationDesc& desc) const {
  auto computation = std::make_unique<GpuComputation>();

  for (const std::string& command : desc.steps) {
    GpuComputationStep step = compileCommand(buffer, command);
    computation->steps.push_back(step);
  }

  return computation;
}

void GpuExecutor::execute(Buffer& buffer, const Computation& computation) const {
  // TODO
}

}

ExecutorPtr createGpuExecutor(Logger& logger) {
  return std::make_unique<GpuExecutor>(logger);
}

BufferPtr createGpuBuffer() {
  return std::make_unique<GpuBuffer>();
}
