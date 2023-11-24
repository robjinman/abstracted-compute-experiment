#include "cpu_compute.hpp"
#include "gpu_compute.hpp"
#include "logger.hpp"
#include "utils.hpp"

int main() {
  LoggerPtr logger = createStdoutLogger();

  //ExecutorPtr executor = createCpuExecutor(*logger);
  //BufferPtr buffer = createCpuBuffer();

  ExecutorPtr executor = createGpuExecutor(*logger);
  BufferPtr buffer = createGpuBuffer();

  Matrix M({
    { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 },
    { 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6 },
    { 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4 },
    { 2, 1, 0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 7 },
  });
  Vector V({ 7, 2, 4, 3, 8, 7, 5, 2, 4, 6, 1, 2, 3, 4, 5, 6 });
  Vector A(4);
  Vector B({ 4, 3, 2, 1 });
  Vector C(4);

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
  executor->execute(*buffer, *c);

  // [ 642 696 598 706 ]
  logger->info(STR(C));

  return 0;
}
