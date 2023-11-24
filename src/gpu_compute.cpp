#include "gpu_compute.hpp"
#include "types.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include "gpu.hpp"
#include <map>
#include <fstream>
#include <variant>
#include <cstring>

namespace {

struct GpuBufferItem {
  MathObjectType type;
  Triple shape;
  size_t offset;
};

class GpuBuffer : public Buffer {
  public:
    std::vector<double> storage;
    std::map<std::string, GpuBufferItem> items;

    void insert(const std::string& name, Array& item) override;
    void insert(const std::string& name, Array2& item) override;
    void insert(const std::string& name, Array3& item) override;

  private:
    template<class T>
    void insertItem(const std::string& name, T& item);
};

template<class T>
void GpuBuffer::insertItem(const std::string& name, T& item) {
  size_t size = item.storage().size();
  size_t offset = storage.size();
  storage.resize(offset + size);
  memcpy(storage.data() + offset, item.storage().data(), size * sizeof(double));
  item.setDataPtr(storage.data() + offset);
  items.insert({ name, GpuBufferItem{ item.type(), item.shape(), offset } });
}

void GpuBuffer::insert(const std::string& name, Array& item) {
  insertItem(name, item);
}

void GpuBuffer::insert(const std::string& name, Array2& item) {
  insertItem(name, item);
}

void GpuBuffer::insert(const std::string& name, Array3& item) {
  insertItem(name, item);
}

struct GpuComputationStep {
  std::string commands;
  size_t shader;
  size_t numWorkgroups;
};

class GpuComputation : public Computation {
  public:
    std::vector<GpuComputationStep> steps;
};

using GpuComputationPtr = std::unique_ptr<GpuComputation>;

struct ShaderSnippet {
  size_t workgroupSize;
  std::string source;
};

class Token {
  public:
    Token(double value);
    Token(const GpuBufferItem& bufferItem);

    bool isNumeric() const;
    double floatValue() const;
    const GpuBufferItem& bufferItem() const;

  private:
    std::variant<double, GpuBufferItem> m_value;
};

Token::Token(double value)
  : m_value(value) {}

Token::Token(const GpuBufferItem& bufferItem)
  : m_value(bufferItem) {}

bool Token::isNumeric() const {
  return std::holds_alternative<double>(m_value);
}

double Token::floatValue() const {
  return std::get<double>(m_value);
}

const GpuBufferItem& Token::bufferItem() const {
  return std::get<GpuBufferItem>(m_value);
}

bool parseDouble(const std::string& strValue, double& value) {
  std::stringstream ss(strValue);
  ss >> value;
  return !ss.fail() && ss.eof();
}

Token parseToken(const GpuBuffer& buffer, const std::string& strToken) {
  double value = 0;
  if (parseDouble(strToken, value)) {
    return value;
  }
  else {
    return buffer.items.at(strToken);
  }
}

ShaderSnippet compileMultiplyCommand(const GpuBuffer& buffer,
  const std::vector<std::string>& tokens) {

  const GpuBufferItem& returnVal = buffer.items.at(tokens[0]);
  const std::string& functionName = tokens[1];
  
  ASSERT(functionName == "multiply");
  ASSERT(tokens.size() == 4);

  ShaderSnippet snippet;

  Token arg1 = parseToken(buffer, tokens[2]);
  Token arg2 = parseToken(buffer, tokens[3]);

  if (arg1.isNumeric()) {
    EXCEPTION("No function 'multiply' matching argument types");
  }
  else if (arg1.bufferItem().type == MathObjectType::Array) {
    if (arg2.isNumeric()) {
      size_t rOffset = returnVal.offset;
      size_t vOffset = arg1.bufferItem().offset;
      size_t vSize = arg1.bufferItem().shape[0];
      double x = arg2.floatValue();

      snippet.source = STR("vecScalarMultiply(" << vOffset << ", " << vSize << ", " << x << ", "
        << rOffset << ");");

      snippet.workgroupSize = vSize;
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
      size_t rOffset = returnVal.offset;
      size_t mOffset = arg1.bufferItem().offset;
      size_t mCols = arg1.bufferItem().shape[0];
      size_t mRows = arg1.bufferItem().shape[1];
      size_t vOffset = arg2.bufferItem().offset;
      size_t vSize = arg2.bufferItem().shape[0];

      ASSERT_MSG(mCols == vSize, "Cannot multiply a " << mCols
        << "-column matrix with a vector of size " << vSize);

      snippet.source = STR("matVecMultiply(" << mOffset << ", " << mCols << ", " << mRows << ", "
        << vOffset << ", " << vSize << ", " << rOffset << ");");

      snippet.workgroupSize = mRows;
    }
    else {
      EXCEPTION("No function 'multiply' matching argument types");
    }
  }
  else {
    EXCEPTION("No function 'multiply' matching argument types");
  }

  return snippet;
}

ShaderSnippet compileAddCommand(const GpuBuffer& buffer, const std::vector<std::string>& tokens) {
  const GpuBufferItem& returnVal = buffer.items.at(tokens[0]);
  const std::string& functionName = tokens[1];
  
  ASSERT(functionName == "add");
  ASSERT(tokens.size() == 4);

  ShaderSnippet snippet;

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
      size_t rOffset = returnVal.offset;
      size_t aOffset = arg1.bufferItem().offset;
      size_t aSize = arg1.bufferItem().shape[0];
      size_t bOffset = arg2.bufferItem().offset;
      size_t bSize = arg2.bufferItem().shape[0];

      ASSERT_MSG(aSize == bSize, "Cannot add vectors of sizes " << aSize << " and " << bSize);

      snippet.source = STR("vecVecAdd(" << aOffset << ", " << aSize << ", " << bOffset << ", "
        << bSize << rOffset << ");");

      snippet.workgroupSize = aSize;
    }
    else {
      EXCEPTION("No function 'add' matching argument types");
    }
  }
  else {
    EXCEPTION("No function 'add' matching argument types");
  }

  return snippet;
}

ShaderSnippet compileCommand(const Buffer& buf, const std::string& command) {
  const auto& buffer = dynamic_cast<const GpuBuffer&>(buf);

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

class GpuExecutor : public Executor {
  public:
    GpuExecutor(Logger& logger);
  
    ComputationPtr compile(const Buffer& buffer, const ComputationDesc& desc) const override;
    void execute(Buffer& buffer, const Computation& computation) const override;

  private:
    GpuComputationStep compileStep(std::vector<std::string>& snippets, size_t workgroupSize) const;

    Logger& m_logger;
    GpuPtr m_gpu;
};

GpuExecutor::GpuExecutor(Logger& logger)
  : m_logger(logger)
  , m_gpu(createGpu()) {}

GpuComputationStep GpuExecutor::compileStep(std::vector<std::string>& snippets,
  size_t workgroupSize) const {

  m_logger.info("Compiling...");

  std::ifstream fin("data/functions.glsl");
  std::stringstream ss;

  ss << "#version 450" << std::endl << std::endl;
  ss << "layout (local_size_x = " << workgroupSize << ") in;" << std::endl << std::endl;

  std::string line;
  while (std::getline(fin, line)) {
    ss << line << std::endl;
  }

  ss << std::endl;
  ss << "void main() {" << std::endl;

  for (const std::string& snippet : snippets) {
    ss << snippet << std::endl;
  }

  ss << "}" << std::endl;

  GpuComputationStep step;
  m_logger.info(ss.str()); // TODO
  step.shader = m_gpu->compileShader(ss.str());
  step.numWorkgroups = 1; // TODO
  // TODO: Set step.commands

  return step;
}

ComputationPtr GpuExecutor::compile(const Buffer& buffer, const ComputationDesc& desc) const {
  auto computation = std::make_unique<GpuComputation>();

  std::vector<std::string> snippets;
  size_t currentWorkgroupSize = 0;

  for (const std::string& command : desc.steps) {
    ShaderSnippet snippet = compileCommand(buffer, command);

    if (snippet.workgroupSize == currentWorkgroupSize || snippets.empty()) {
      m_logger.info("A");
      snippets.push_back(snippet.source);
      currentWorkgroupSize = snippet.workgroupSize;
    }
    else {
      m_logger.info("B");
      computation->steps.push_back(compileStep(snippets, currentWorkgroupSize));
      snippets.clear();
      snippets.push_back(snippet.source);
      currentWorkgroupSize = 0;
    }
  }

  if (!snippets.empty()) {
      m_logger.info("C");
    ASSERT(currentWorkgroupSize != 0);
    computation->steps.push_back(compileStep(snippets, currentWorkgroupSize));
  }

  return computation;
}

void GpuExecutor::execute(Buffer& buf, const Computation& computation) const {
  auto& buffer = dynamic_cast<GpuBuffer&>(buf);
  m_gpu->submitBuffer(buffer.storage.data(), buffer.storage.size() * sizeof(double));

  const auto& c = dynamic_cast<const GpuComputation&>(computation);
  for (const auto& step : c.steps) {
#ifndef NDEBUG
    m_logger.info(STR("Executing commands: " << step.commands));
#endif
    m_gpu->executeShader(step.shader, step.numWorkgroups);
  }

  m_gpu->retrieveBuffer(buffer.storage.data());
}

}

ExecutorPtr createGpuExecutor(Logger& logger) {
  return std::make_unique<GpuExecutor>(logger);
}

BufferPtr createGpuBuffer() {
  return std::make_unique<GpuBuffer>();
}
