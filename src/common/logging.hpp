#pragma once

#include <triton/core/tritonserver.h>
#include <cstdio>

#ifndef LOG_DISABLE
// 生产环境在编译时定义 LOG_DISABLE 可关闭所有日志

#define LOG_VERBOSE(fmt, ...)                                                 \
    do {                                                                      \
        char _log_buf[512];                                                   \
        snprintf(_log_buf, sizeof(_log_buf), fmt, ##__VA_ARGS__);            \
        TRITONSERVER_LogMessage(TRITONSERVER_LOG_VERBOSE, __FILE__,          \
                                __LINE__, _log_buf);                          \
    } while (0)

#define LOG_INFO(fmt, ...)                                                    \
    do {                                                                      \
        char _log_buf[512];                                                   \
        snprintf(_log_buf, sizeof(_log_buf), fmt, ##__VA_ARGS__);            \
        TRITONSERVER_LogMessage(TRITONSERVER_LOG_INFO, __FILE__,             \
                                __LINE__, _log_buf);                          \
    } while (0)

#define LOG_WARN(fmt, ...)                                                    \
    do {                                                                      \
        char _log_buf[512];                                                   \
        snprintf(_log_buf, sizeof(_log_buf), fmt, ##__VA_ARGS__);            \
        TRITONSERVER_LogMessage(TRITONSERVER_LOG_WARN, __FILE__,             \
                                __LINE__, _log_buf);                          \
    } while (0)

#define LOG_ERROR(fmt, ...)                                                   \
    do {                                                                      \
        char _log_buf[512];                                                   \
        snprintf(_log_buf, sizeof(_log_buf), fmt, ##__VA_ARGS__);            \
        TRITONSERVER_LogMessage(TRITONSERVER_LOG_ERROR, __FILE__,            \
                                __LINE__, _log_buf);                          \
    } while (0)

#else
#define LOG_VERBOSE(fmt, ...) ((void)0)
#define LOG_INFO(fmt, ...)    ((void)0)
#define LOG_WARN(fmt, ...)    ((void)0)
#define LOG_ERROR(fmt, ...)   ((void)0)
#endif
