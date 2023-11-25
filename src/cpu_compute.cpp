#include "cpu_compute.hpp"
#include "exception.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include <variant>
#include <map>
#include <functional>

namespace {

using MathObjectPtr = std::variant<ArrayPtr, Array2Ptr, Array3Ptr>;

class CpuBuffer : public Buffer {
  public:
    struct Entry {
      size_t index;
      MathObjectType type;
    };

    void insert(const std::string& name, Array& object) override;
    void insert(const std::string& name, Array2& object) override;
    void insert(const std::string& name, Array3& object) override;

    std::vector<MathObjectPtr> items;
    std::map<std::string, Entry> entries;
};

void CpuBuffer::insert(const std::string& name, Array& item) {
  size_t index = items.size();
  items.push_back(Array::createShallow(item.storage()));
  entries[name] = Entry{ index, MathObjectType::Array };
}

void CpuBuffer::insert(const std::string& name, Array2& item) {
  size_t index = items.size();
  items.push_back(Array2::createShallow(item.storage(), item.cols(), item.rows()));
  entries[name] = Entry{ index, MathObjectType::Array2 };
}

void CpuBuffer::insert(const std::string& name, Array3& item) {
  size_t index = items.size();
  items.push_back(Array3::createShallow(item.storage(), item.W(), item.H(), item.D()));
  entries[name] = Entry{ index, MathObjectType::Array3 };
}

using CpuComputationStepFn = std::function<void()>;

struct CpuComputationStep {
  std::string command;
  CpuComputationStepFn function;
};

class CpuComputation : public Computation {
  public:
    std::vector<CpuComputationStep> steps;
};

using CpuComputationPtr = std::unique_ptr<CpuComputation>;

class CpuExecutor : public Executor {
  public:
    CpuExecutor(Logger& logger);
  
    ComputationPtr compile(const Buffer& buffer, const ComputationDesc& desc) const override;
    void execute(Buffer& buffer, const Computation& computation,
      size_t iterations = 1) const override;

  private:
    Logger& m_logger;
};

class Token {
  public:
    Token(netfloat_t value);
    Token(const CpuBuffer::Entry& bufferEntry);

    bool isNumeric() const;
    netfloat_t floatValue() const;
    const CpuBuffer::Entry& bufferEntry() const;

  private:
    std::variant<netfloat_t, CpuBuffer::Entry> m_value;
};

Token::Token(netfloat_t value)
  : m_value(value) {}

Token::Token(const CpuBuffer::Entry& bufferEntry)
  : m_value(bufferEntry) {}

bool Token::isNumeric() const {
  return std::holds_alternative<netfloat_t>(m_value);
}

netfloat_t Token::floatValue() const {
  return std::get<netfloat_t>(m_value);
}

const CpuBuffer::Entry& Token::bufferEntry() const {
  return std::get<CpuBuffer::Entry>(m_value);
}

bool parsenetfloat_t(const std::string& strValue, netfloat_t& value) {
  std::stringstream ss(strValue);
  ss >> value;
  return !ss.fail() && ss.eof();
}

Token parseToken(const CpuBuffer& buffer, const std::string& strToken) {
  netfloat_t value = 0;
  if (parsenetfloat_t(strToken, value)) {
    return value;
  }
  else {
    return buffer.entries.at(strToken);
  }
}

CpuComputationStep compileMultiplyCommand(const CpuBuffer& buffer,
  const std::vector<std::string>& tokens) {

  const CpuBuffer::Entry& returnVal = buffer.entries.at(tokens[0]);
  const std::string& functionName = tokens[1];
  
  ASSERT(functionName == "multiply");
  ASSERT(tokens.size() == 4);

  CpuComputationStep step;
  step.command = functionName;

  Token arg1 = parseToken(buffer, tokens[2]);
  Token arg2 = parseToken(buffer, tokens[3]);

  if (arg1.isNumeric()) {
    EXCEPTION("No function 'multiply' matching argument types");
  }
  else if (arg1.bufferEntry().type == MathObjectType::Array) {
    if (arg2.isNumeric()) {
      Vector& R = *(std::get<VectorPtr>(buffer.items[returnVal.index]));
      Vector& V = *(std::get<VectorPtr>(buffer.items[arg1.bufferEntry().index]));
      netfloat_t x = arg2.floatValue();

      step.function = [&R, &V, x]() {
        R = V * x;
      };
    }
    else {
      EXCEPTION("No function 'multiply' matching argument types");
    }
  }
  else if (arg1.bufferEntry().type == MathObjectType::Array2) {
    if (arg2.isNumeric()) {
      EXCEPTION("No function 'multiply' matching argument types");
    }
    else if (arg2.bufferEntry().type == MathObjectType::Array) {
      Vector& R = *(std::get<VectorPtr>(buffer.items[returnVal.index]));
      Matrix& M = *(std::get<MatrixPtr>(buffer.items[arg1.bufferEntry().index]));
      Vector& V = *(std::get<VectorPtr>(buffer.items[arg2.bufferEntry().index]));

      step.function = [&R, &M, &V]() {
        R = M * V;
      };
    }
    else {
      EXCEPTION("No function 'multiply' matching argument types");
    }
  }
  else {
    EXCEPTION("No function 'multiply' matching argument types");
  }
  
  return step;
}

CpuComputationStep compileAddCommand(const CpuBuffer& buffer,
  const std::vector<std::string>& tokens) {

  const CpuBuffer::Entry& returnVal = buffer.entries.at(tokens[0]);
  const std::string& functionName = tokens[1];
  
  ASSERT(functionName == "add");
  ASSERT(tokens.size() == 4);

  CpuComputationStep step;
  step.command = functionName;

  Token arg1 = parseToken(buffer, tokens[2]);
  Token arg2 = parseToken(buffer, tokens[3]);

  if (arg1.isNumeric()) {
    EXCEPTION("No function 'add' matching argument types");
  }
  else if (arg1.bufferEntry().type == MathObjectType::Array) {
    if (arg2.isNumeric()) {
      EXCEPTION("No function 'add' matching argument types");
    }
    else if (arg2.bufferEntry().type == MathObjectType::Array) {
      Vector& R = *(std::get<VectorPtr>(buffer.items[returnVal.index]));
      Vector& A = *(std::get<VectorPtr>(buffer.items[arg1.bufferEntry().index]));
      Vector& B = *(std::get<VectorPtr>(buffer.items[arg2.bufferEntry().index]));

      step.function = [&R, &A, &B]() {
        R = A + B;
      };
    }
    else {
      EXCEPTION("No function 'add' matching argument types");
    }
  }
  else {
    EXCEPTION("No function 'add' matching argument types");
  }
  
  return step;
}

CpuComputationStep compileCommand(const Buffer& buf, const std::string& command) {
  const auto& buffer = dynamic_cast<const CpuBuffer&>(buf);

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

CpuExecutor::CpuExecutor(Logger& logger)
  : m_logger(logger) {}

ComputationPtr CpuExecutor::compile(const Buffer& buffer, const ComputationDesc& desc) const {
  auto computation = std::make_unique<CpuComputation>();

  for (const std::string& command : desc.steps) {
    CpuComputationStep step = compileCommand(buffer, command);
    computation->steps.push_back(step);
  }

  return computation;
}

void CpuExecutor::execute(Buffer&, const Computation& computation, size_t iterations) const {
  const auto& c = dynamic_cast<const CpuComputation&>(computation);
  for (size_t i = 0; i < iterations; ++i) {
    for (const auto& step : c.steps) {
#ifndef NDEBUG
      m_logger.info(STR("Executing command: " << step.command));
#endif
      step.function();
    }
  }
}

}

ExecutorPtr createCpuExecutor(Logger& logger) {
  return std::make_unique<CpuExecutor>(logger);
}

BufferPtr createCpuBuffer() {
  return std::make_unique<CpuBuffer>();
}
