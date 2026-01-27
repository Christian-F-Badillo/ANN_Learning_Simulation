#pragma once

#include <atomic>
#include <iostream>
#include <mutex>
#include <string>

// --- COLORES ANSI ---
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BOLD "\033[1m"

// --- GLOBALES THREAD-SAFE ---
inline std::atomic<int> g_failed_tests{0};
inline std::atomic<int> g_total_asserts{0};
inline std::mutex g_io_mutex;

template <typename T>
void print_failure(const char *file, int line, const std::string &msg,
                   const T &expected, const T &actual) {
  std::lock_guard<std::mutex> lock(g_io_mutex);

  std::cerr << COLOR_RED << "  [FAIL] " << COLOR_RESET << file << ":" << line
            << " | " << msg << "\n"
            << "         Esperado: " << expected << "\n"
            << "         Obtenido: " << actual << std::endl;
  g_failed_tests++;
}

// --- MACROS ---
#define TEST_CASE(name)                                                        \
  do {                                                                         \
    std::lock_guard<std::mutex> lock(g_io_mutex);                              \
    std::cout << COLOR_BOLD << "\nRunning Test: " << name << "..."             \
              << COLOR_RESET << std::endl;                                     \
  } while (0)

#define ASSERT_EQ(actual, expected)                                            \
  do {                                                                         \
    g_total_asserts++;                                                         \
    if ((actual) != (expected)) {                                              \
      print_failure(__FILE__, __LINE__, "Equality check failed", (expected),   \
                    (actual));                                                 \
    }                                                                          \
  } while (0)

#define ASSERT_ALMOST_EQ(actual, expected)                                     \
  do {                                                                         \
    g_total_asserts++;                                                         \
    /* Usamos 1e-4 como tolerancia estándar para Deep Learning (float) */      \
    if (std::abs((actual) - (expected)) > 1e-4) {                              \
      print_failure(__FILE__, __LINE__, "Float precision check failed",        \
                    (expected), (actual));                                     \
    }                                                                          \
  } while (0)

#define ASSERT_THROWS(expression, exception_type)                              \
  do {                                                                         \
    g_total_asserts++;                                                         \
    bool threw = false;                                                        \
    try {                                                                      \
      expression;                                                              \
    } catch (const exception_type &) {                                         \
      threw = true;                                                            \
    } catch (...) {                                                            \
      std::lock_guard<std::mutex> lock(g_io_mutex);                            \
      std::cerr << COLOR_RED << "  [FAIL] " << COLOR_RESET << __FILE__ << ":"  \
                << __LINE__ << " | Lanzó una excepción incorrecta."            \
                << std::endl;                                                  \
      g_failed_tests++;                                                        \
      break;                                                                   \
    }                                                                          \
    if (!threw) {                                                              \
      std::lock_guard<std::mutex> lock(g_io_mutex);                            \
      std::cerr << COLOR_RED << "  [FAIL] " << COLOR_RESET << __FILE__ << ":"  \
                << __LINE__ << " | Se esperaba " << #exception_type            \
                << " pero no lanzó nada." << std::endl;                        \
      g_failed_tests++;                                                        \
    }                                                                          \
  } while (0)

inline int run_test_summary() {
  std::lock_guard<std::mutex> lock(g_io_mutex);
  if (g_failed_tests == 0) {
    std::cout << "\n"
              << COLOR_GREEN
              << "==========================================" << std::endl;
    std::cout << "  EXITO: Todos los tests (" << g_total_asserts
              << ") pasaron correctamente." << std::endl;
    std::cout << "==========================================" << COLOR_RESET
              << std::endl;
    return 0;
  } else {
    std::cout << "\n"
              << COLOR_RED
              << "==========================================" << std::endl;
    std::cout << "  FALLO: " << g_failed_tests << " tests fallaron de "
              << g_total_asserts << "." << std::endl;
    std::cout << "==========================================" << COLOR_RESET
              << std::endl;
    return 1;
  }
}
