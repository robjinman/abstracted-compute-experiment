#include "cpu_compute.hpp"
#include "gpu_compute.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include <chrono>

using std::chrono::duration_cast;

struct InputData {
  Matrix M;
  Vector V;
  Vector B;
};

void runBenchmark(Logger& logger, const InputData& data, bool gpu) {
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

  Matrix M = data.M;
  Vector V = data.V;
  Vector B = data.B;
  Vector A(B.size());
  Vector C(B.size());

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

  Timer timer;
  const size_t iterations = 100;

  timer.start();
  executor->execute(*buffer, *c, iterations);
  auto elapsed = timer.stop();

  logger.info(STR("Running time: " << elapsed / 1000.0 << " milliseconds"));

  //logger.info(STR(C));
}

int main() {
  LoggerPtr logger = createStdoutLogger();

  InputData data{
    Matrix(1024, 1024),
    Vector(1024),
    Vector(1024)
  };
  data.M.fill(1);
  data.V.fill(1);
  data.B.fill(1);
  //data.M.randomize(1.0);
  //data.V.randomize(1.0);
  //data.B.randomize(1.0);

  logger->info("Running CPU benchmark...");
  runBenchmark(*logger, data, false);

  logger->info("Running GPU benchmark...");
  runBenchmark(*logger, data, true);

  return 0;
}
