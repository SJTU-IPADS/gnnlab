/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef SAMGRAPH_LOGGING_H
#define SAMGRAPH_LOGGING_H

#include <sstream>
#include <string>

namespace samgraph {
namespace common {

enum class LogLevel { TRACE, DEBUG, INFO, WARNING, ERROR, FATAL };

#define LOG_LEVELS "TDIWEF"

// Always-on checking
#define CHECK(x) \
  if (!(x))      \
  common::LogMessageFatal(__FILE__, __LINE__) << "Check failed: " #x << ' '

#define CHECK_LT(x, y) CHECK((x) < (y))
#define CHECK_GT(x, y) CHECK((x) > (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_GE(x, y) CHECK((x) >= (y))
#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_NE(x, y) CHECK((x) != (y))
#define CHECK_NOTNULL(x)                                     \
  ((x) == NULL ? common::LogMessageFatal(__FILE__, __LINE__) \
                     << "Check  notnull: " #x << ' ',        \
   (x) : (x))  // NOLINT(*)

/*!
 * \brief Protected CUDA call.
 * \param func Expression to call.
 *
 * It checks for CUDA errors after invocation of the expression.
 */
#define CUDA_CALL(func)                                      \
  {                                                          \
    cudaError_t e = (func);                                  \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                \
  }

/*!
 * \brief Protected CUSPARSE call.
 */
#define CUSPARSE_CALL(func)                           \
  {                                                   \
    cusparseStatus_t e = (func);                      \
    CHECK(e == CUSPARSE_STATUS_SUCCESS)               \
        << "CUSPARSE: " << cusparseGetErrorString(e); \
  }

/*
 * \brief Protected NCCL call.
 */
#define NCCLCHECK(cmd)                                                  \
  {                                                                     \
    ncclResult_t r = (cmd);                                             \
    CHECK(r == ncclSuccess) << "NCCL error: " << ncclGetErrorString(r); \
  }

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, LogLevel severity);
  ~LogMessage();

 protected:
  void GenerateLogMessage(bool log_time);

 private:
  const char* fname_;
  int line_;
  LogLevel severity_;
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal();
};

#define _LOG_TRACE \
  common::LogMessage(__FILE__, __LINE__, common::LogLevel::TRACE)
#define _LOG_DEBUG \
  common::LogMessage(__FILE__, __LINE__, common::LogLevel::DEBUG)
#define _LOG_INFO common::LogMessage(__FILE__, __LINE__, common::LogLevel::INFO)
#define _LOG_WARNING \
  common::LogMessage(__FILE__, __LINE__, common::LogLevel::WARNING)
#define _LOG_ERROR \
  common::LogMessage(__FILE__, __LINE__, common::LogLevel::ERROR)
#define _LOG_FATAL common::LogMessageFatal(__FILE__, __LINE__)

#define _LOG(severity) _LOG_##severity

#define _LOG_RANK(severity, rank) _LOG_##severity << "[" << rank << "]: "

#define GET_LOG(_1, _2, NAME, ...) NAME
#define LOG(...) GET_LOG(__VA_ARGS__, _LOG_RANK, _LOG)(__VA_ARGS__)

#define DEBUG_PREFIX "\033[1;33mDEBUG: \033[0m"
#define WARNING_PREFIX "\033[38;5;215mWARNING: \033[0m"

LogLevel MinLogLevelFromEnv();
bool LogTimeFromEnv();

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_LOGGING_H
