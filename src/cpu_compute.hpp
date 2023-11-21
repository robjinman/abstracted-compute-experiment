#pragma once

#include "compute.hpp"

class Logger;
ExecutorPtr createCpuExecutor(Logger& logger);
