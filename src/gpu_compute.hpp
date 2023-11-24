#pragma once

#include "compute.hpp"

class Logger;
ExecutorPtr createGpuExecutor(Logger& logger);
BufferPtr createGpuBuffer();
