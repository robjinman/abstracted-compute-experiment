#include "gpu_compute.hpp"

namespace {

class GpuComputation : public Computation {
  public:
    // TODO
};

using GpuComputationPtr = std::unique_ptr<GpuComputation>;

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
  // TODO
}

void GpuExecutor::execute(Buffer& buffer, const Computation& computation) const {
  // TODO
}

}

ExecutorPtr createGpuExecutor(Logger& logger) {
  return std::make_unique<GpuExecutor>(logger);
}
