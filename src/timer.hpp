#pragma once

#include <cstdint>
#include <chrono>

class Timer {
  public:
    inline void start();
    inline int64_t stop() const;

  private:
    std::chrono::high_resolution_clock::time_point m_startTime;
};

void Timer::start() {
  m_startTime = std::chrono::high_resolution_clock::now();
}

int64_t Timer::stop() const {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(now - m_startTime).count();
}
