#pragma once

#include <string>
#include <memory>
#include <vector>

class Gpu {
  public:
    virtual void submitBuffer(const void* buffer, size_t bufferSize) = 0;
    virtual void updateBuffer(const void* data) = 0;
    virtual void executeShader(size_t shaderIndex, size_t numWorkgroups) = 0;
    virtual void retrieveBuffer(void* data) = 0;

    virtual ~Gpu() {}
};

using GpuPtr = std::unique_ptr<Gpu>;

GpuPtr createGpu(const std::vector<std::string>& shaderSources);
