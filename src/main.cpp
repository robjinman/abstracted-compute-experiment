#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include <map>
#include <variant>
#include <vector>
#include "math.hpp"

#define STR(x) (std::stringstream("") << x).str()

struct BufferItem {
  MathObjectType type;
  size_t offset;
};

struct Buffer {
  std::vector<uint8_t> storage;
  std::map<std::string, BufferItem> items;

  void insert(const std::string& name, Array& item);
  void insert(const std::string& name, Array2& item);
  void insert(const std::string& name, Array3& item);
};

void Buffer::insert(const std::string& name, Array& item) {
  size_t offset = storage.size();
  storage.resize(offset + item.serializedSize());
  item.serialize(storage.data() + offset);
  items.insert({ name, BufferItem{ MathObjectType::Array, offset } });
}

void Buffer::insert(const std::string& name, Array2& item) {
  size_t offset = storage.size();
  storage.resize(offset + item.serializedSize());
  item.serialize(storage.data() + offset);
  items.insert({ name, BufferItem{ MathObjectType::Array2, offset } });
}

void Buffer::insert(const std::string& name, Array3& item) {
  size_t offset = storage.size();
  storage.resize(offset + item.serializedSize());
  item.serialize(storage.data() + offset);
  items.insert({ name, BufferItem{ MathObjectType::Array3, offset } });
}

struct ComputationDesc {
  std::vector<std::string> steps;
  
  void chain(const ComputationDesc& c);
};

void ComputationDesc::chain(const ComputationDesc& c) {
  steps.insert(steps.end(), c.steps.begin(), c.steps.end());
}

class Computation {
  public:
    virtual ~Computation() = 0;
};

using ComputationPtr = std::unique_ptr<Computation>;

Computation::~Computation() {}

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

class Executor {
  public:
    virtual ComputationPtr compile(const Buffer& buffer, const ComputationDesc& desc) const = 0;
    virtual void execute(Buffer& buffer, const Computation& computation) const = 0;

    virtual ~Executor() {}
};

using ExecutorPtr = std::unique_ptr<Executor>;

class CpuExecutor : public Executor {
  public:
    ComputationPtr compile(const Buffer& buffer, const ComputationDesc& desc) const override;
    void execute(Buffer& buffer, const Computation& computation) const override;
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

CpuComputationStep parseCommand(const Buffer& buffer, const std::string& command) {
  std::vector<std::string> tokens = tokenizeCommand(command);

  ASSERT(tokens.size() >= 2);
  const BufferItem& returnVal = buffer.items.at(tokens[0]);
  const std::string& functionName = tokens[1];

  CpuComputationStep step;
  step.command = command;

  if (functionName == "multiply") {
    ASSERT(tokens.size() == 4);
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
  }
  else if (functionName == "add") {
    ASSERT(tokens.size() == 4);
    Token arg1 = parseToken(buffer, tokens[2]);
    Token arg2 = parseToken(buffer, tokens[3]);
  
    if (arg1.isNumeric()) {
      EXCEPTION("No function 'multiply' matching argument types");
    }
    else if (arg1.bufferItem().type == MathObjectType::Array) {
      if (arg2.isNumeric()) {
        EXCEPTION("No function 'multiply' matching argument types");
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
  }
  else {
    EXCEPTION("Function '" << functionName << "' not recognised");
  }

  return step;
}

ComputationPtr CpuExecutor::compile(const Buffer& buffer, const ComputationDesc& desc) const {
  auto computation = std::make_unique<CpuComputation>();

  for (const std::string& command : desc.steps) {
    CpuComputationStep step = parseCommand(buffer, command);
    computation->steps.push_back(step);
  }

  return computation;
}

void CpuExecutor::execute(Buffer& buffer, const Computation& computation) const {
  const auto& c = dynamic_cast<const CpuComputation&>(computation);
  for (const auto& step : c.steps) {
#ifndef NDEBUG
    std::cout << "Executing command: " << step.command << std::endl;
#endif
    step.function(buffer.storage.data());
  }
}

int main() {
  ExecutorPtr executor = std::make_unique<CpuExecutor>();

  Matrix M({
    { 1, 2, 3, 4 },
    { 5, 6, 7, 8 },
    { 9, 0, 1, 2 }
  });
  Vector V({7, 2, 4, 3});
  Vector A(3);
  Vector B({4, 3, 2});
  Vector C(3);

  Buffer buffer; // TODO: Do this efficiently
  buffer.insert("M", M);
  buffer.insert("V", V);
  buffer.insert("A", A);
  buffer.insert("B", B);
  buffer.insert("C", C);

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

  ComputationPtr c = executor->compile(buffer, comp1);
  executor->execute(buffer, *c);

  std::cout << C;

  return 0;
}

