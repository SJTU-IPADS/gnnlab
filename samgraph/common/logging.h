#ifndef SAMGRAPH_LOGGING_H
#define SAMGRAPH_LOGGING_H

#include <sstream>
#include <string>

namespace samgraph {
namespace common {

enum class LogLevel { TRACE, DEBUG, INFO, WARNING, ERROR, FATAL };

#define LOG_LEVELS "TDIWEF"

// Always-on checking
#define SAM_CHECK(x) \
  if (!(x))          \
  common::LogMessageFatal(__FILE__, __LINE__) << "Check failed: " #x << ' '

#define SAM_CHECK_LT(x, y) SAM_CHECK((x) < (y))
#define SAM_CHECK_GT(x, y) SAM_CHECK((x) > (y))
#define SAM_CHECK_LE(x, y) SAM_CHECK((x) <= (y))
#define SAM_CHECK_GE(x, y) SAM_CHECK((x) >= (y))
#define SAM_CHECK_EQ(x, y) SAM_CHECK((x) == (y))
#define SAM_CHECK_NE(x, y) SAM_CHECK((x) != (y))
#define SAM_CHECK_NOTNULL(x)                                 \
  ((x) == NULL ? common::LogMessageFatal(__FILE__, __LINE__) \
                     << "Check  notnull: " #x << ' ',        \
   (x) : (x))  // NOLINT(*)

/*!
 * \brief Protected CUDA call.
 * \param func Expression to call.
 *
 * It checks for CUDA errors after invocation of the expression.
 */
#define CUDA_CALL(func)                                          \
  {                                                              \
    cudaError_t e = (func);                                      \
    SAM_CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                    \
  }

/*!
 * \brief Protected CUSPARSE call.
 */
#define CUSPARSE_CALL(func)                                      \
  {                                                              \
    cusparseStatus_t e = (func);                                 \
    SAM_CHECK(e == CUSPARSE_STATUS_SUCCESS)                      \
        << "CUSPARSE: " << cusparseGetErrorString(e);            \
  }

/*
 * \brief Protected NCCL call.
 */
#define NCCLCHECK(cmd)                                                      \
  {                                                                         \
    ncclResult_t r = (cmd);                                                 \
    SAM_CHECK(r == ncclSuccess) << "NCCL error: " << ncclGetErrorString(r); \
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

#define _SAM_LOG_TRACE LogMessage(__FILE__, __LINE__, LogLevel::TRACE)
#define _SAM_LOG_DEBUG LogMessage(__FILE__, __LINE__, LogLevel::DEBUG)
#define _SAM_LOG_INFO LogMessage(__FILE__, __LINE__, LogLevel::INFO)
#define _SAM_LOG_WARNING LogMessage(__FILE__, __LINE__, LogLevel::WARNING)
#define _SAM_LOG_ERROR LogMessage(__FILE__, __LINE__, LogLevel::ERROR)
#define _SAM_LOG_FATAL LogMessageFatal(__FILE__, __LINE__)

#define _LOG(severity) _SAM_LOG_##severity

#define _LOG_RANK(severity, rank) _SAM_LOG_##severity << "[" << rank << "]: "

#define GET_LOG(_1, _2, NAME, ...) NAME
#define SAM_LOG(...) GET_LOG(__VA_ARGS__, _LOG_RANK, _LOG)(__VA_ARGS__)

LogLevel MinLogLevelFromEnv();
bool LogTimeFromEnv();

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_LOGGING_H
