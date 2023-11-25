#include "cpu_compute.hpp"
#include "gpu_compute.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include <chrono>

using std::chrono::duration_cast;

long long runBenchmark(Logger& logger, bool gpu) {
  ExecutorPtr executor;
  BufferPtr buffer;

  if (gpu) {
    executor = createGpuExecutor(logger);
    buffer = createGpuBuffer();
  }
  else {
    executor = createCpuExecutor(logger);
    buffer = createCpuBuffer();
  }

  Matrix M(64, 32);
  Vector V(64);
  Vector A(32);
  Vector B(32);
  Vector C(32);

  M.randomize(10.0);
  V.randomize(10.0);
  B.randomize(10.0);

  buffer->insert("M", M);
  buffer->insert("V", V);
  buffer->insert("A", A);
  buffer->insert("B", B);
  buffer->insert("C", C);

  ComputationDesc comp1;
  comp1.steps = {
    "A = multiply M V",
    "C = add A B"
  };

  ComputationDesc comp2;
  comp2.steps = {
    "C = multiply C 2.0"
  };

  comp1.chain(comp2);

  ComputationPtr c = executor->compile(*buffer, comp1);

  auto t1 = std::chrono::high_resolution_clock::now();

  executor->execute(*buffer, *c);

  auto t2 = std::chrono::high_resolution_clock::now();
  return duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

int main() {
  LoggerPtr logger = createStdoutLogger();

  long long cpuTime = runBenchmark(*logger, false);
  long long gpuTime = runBenchmark(*logger, true);

  logger->info(STR("CPU running time: " << cpuTime << " microseconds"));
  logger->info(STR("GPU running time: " << gpuTime << " microseconds"));

  return 0;
}
