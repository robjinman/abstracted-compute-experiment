#include <variant>
#include <functional>
#include "cpu_compute.hpp"
#include "exception.hpp"
#include "logger.hpp"

namespace {

using CpuComputationStepFn = std::function<void(uint8_t*)>;

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
    void execute(Buffer& buffer, const Computation& computation) const override;
    
  private:
    Logger& m_logger;
};

void matVecMultiply(uint8_t* buffer, size_t outOffset, size_t mOffset, size_t vOffset) {
  ConstMatrixPtr pM = Matrix::deserialize(buffer + mOffset);
  ConstVectorPtr pV = Vector::deserialize(buffer + vOffset);
  VectorPtr pResult = Vector::deserialize(buffer + outOffset);

  const Matrix& M = *pM;
  const Vector& V = *pV;
  Vector& result = *pResult;

  result = M * V;
}

void vecFloatMultiply(uint8_t* buffer, size_t outOffset, size_t vOffset, double x) {
  ConstVectorPtr pV = Vector::deserialize(buffer + vOffset);
  VectorPtr pResult = Vector::deserialize(buffer + outOffset);

  const Vector& V = *pV;
  Vector& result = *pResult;

  result = V * x;
}

void vecAdd(uint8_t* buffer, size_t outOffset, size_t aOffset, size_t bOffset) {
  ConstVectorPtr pA = Vector::deserialize(buffer + aOffset);
  ConstVectorPtr pB = Vector::deserialize(buffer + bOffset);
  VectorPtr pResult = Vector::deserialize(buffer + outOffset);

  const Vector& A = *pA;
  const Vector& B = *pB;
  Vector& result = *pResult;

  result = A + B;
}

void trimLeft(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) {
    return !std::isspace(c);
  }));
}

void trimRight(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c) {
    return !std::isspace(c);
  }).base(), s.end());
}

std::vector<std::string> tokenizeCommand(const std::string& command) {
  std::stringstream ss(command);
  std::vector<std::string> tokens;

  std::string token;
  std::getline(ss, token, '=');

  trimLeft(token);
  trimRight(token);

  ASSERT_MSG(token.length() > 0, STR("Syntax error: " << command));
  tokens.push_back(token);

  while (std::getline(ss, token, ' ')) {
    trimLeft(token);
    trimRight(token);

    if (token.length() > 0) {
      tokens.push_back(token);
    }
  }

  return tokens;
}

class Token {
  public:
    Token(double value);
    Token(const BufferItem& bufferItem);

    bool isNumeric() const;
    double floatValue() const;
    const BufferItem& bufferItem() const;

  private:
    std::variant<double, BufferItem> m_value;
};

Token::Token(double value)
  : m_value(value) {}

Token::Token(const BufferItem& bufferItem)
  : m_value(bufferItem) {}
  
bool Token::isNumeric() const {
  return std::holds_alternative<double>(m_value);
}

double Token::floatValue() const {
  return std::get<double>(m_value);
}

const BufferItem& Token::bufferItem() const {
  return std::get<BufferItem>(m_value);
}

bool parseDouble(const std::string& strValue, double& value) {
  std::stringstream ss(strValue);
  ss >> value;
  return !ss.fail() && ss.eof();
}

Token parseToken(const Buffer& buffer, const std::string& strToken) {
  double value = 0;
  if (parseDouble(strToken, value)) {
    return value;
  }
  else {
    return buffer.items.at(strToken);
  }
}

CpuComputationStep compileMultiplyCommand(const Buffer& buffer,
  const std::vector<std::string>& tokens) {

  const BufferItem& returnVal = buffer.items.at(tokens[0]);
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
  else if (arg1.bufferItem().type == MathObjectType::Array) {
    if (arg2.isNumeric()) {
      size_t outOffset = returnVal.offset;
      double x = arg2.floatValue();
      size_t vOffset = arg1.bufferItem().offset;

      step.function = [=](uint8_t* buf) {
        vecFloatMultiply(buf, outOffset, vOffset, x);
      };
    }
    else {
      EXCEPTION("No function 'multiply' matching argument types");
    }
  }
  else if (arg1.bufferItem().type == MathObjectType::Array2) {
    if (arg2.isNumeric()) {
      EXCEPTION("No function 'multiply' matching argument types");
    }
    else if (arg2.bufferItem().type == MathObjectType::Array) {
      size_t outOffset = returnVal.offset;
      size_t aOffset = arg1.bufferItem().offset;
      size_t bOffset = arg2.bufferItem().offset;

      step.function = [=](uint8_t* buf) {
        matVecMultiply(buf, outOffset, aOffset, bOffset);
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

CpuComputationStep compileAddCommand(const Buffer& buffer,
  const std::vector<std::string>& tokens) {

  const BufferItem& returnVal = buffer.items.at(tokens[0]);
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
  else if (arg1.bufferItem().type == MathObjectType::Array) {
    if (arg2.isNumeric()) {
      EXCEPTION("No function 'add' matching argument types");
    }
    else if (arg2.bufferItem().type == MathObjectType::Array) {
      size_t outOffset = returnVal.offset;
      size_t aOffset = arg1.bufferItem().offset;
      size_t bOffset = arg2.bufferItem().offset;

      step.function = [=](uint8_t* buf) {
        vecAdd(buf, outOffset, aOffset, bOffset);
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

CpuComputationStep compileCommand(const Buffer& buffer, const std::string& command) {
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

void CpuExecutor::execute(Buffer& buffer, const Computation& computation) const {
  const auto& c = dynamic_cast<const CpuComputation&>(computation);
  for (const auto& step : c.steps) {
#ifndef NDEBUG
    m_logger.info(STR("Executing command: " << step.command));
#endif
    step.function(buffer.storage.data());
  }
}

}

ExecutorPtr createCpuExecutor(Logger& logger) {
  return std::make_unique<CpuExecutor>(logger);
}

