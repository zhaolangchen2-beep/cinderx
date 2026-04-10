# RuntimeTests/CMakeLists.txt

file(GLOB RUNTIME_TEST_CASE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp
)

set(RUNTIME_TEST_SUPPORT_SOURCES
  ${PROJECT_SOURCE_DIR}/_cinderx-lib.cpp
  ${PROJECT_SOURCE_DIR}/async_lazy_value.cpp
  ${PROJECT_SOURCE_DIR}/module_state.cpp
  ${PROJECT_SOURCE_DIR}/module_c_state.cpp
  ${PROJECT_SOURCE_DIR}/python_runtime.cpp
)

set(RUNTIME_TEST_COMMON_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/fixtures.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/testutil.cpp
)

add_library(runtime-tests-support STATIC
  ${RUNTIME_TEST_SUPPORT_SOURCES}
)

target_link_libraries(
  runtime-tests-support
  PUBLIC
  Python::Python
  borrowed
  cached-properties
  common
  immortalize
  interpreter
  jit
  parallel-gc
  static-python
  asmjit::asmjit
  fmt::fmt
  ZLIB::ZLIB
)

target_include_directories(
  runtime-tests-support
  PUBLIC
  ${PROJECT_SOURCE_DIR}/..
  ${GENERATED_HEADER_DIR}
  ${Python_INCLUDE_DIRS}
)

find_package(Threads REQUIRED)

add_executable(RuntimeTests
  ${RUNTIME_TEST_CASE_SOURCES}
  ${RUNTIME_TEST_COMMON_SOURCES}
)

target_link_libraries(
  RuntimeTests
  PRIVATE
  runtime-tests-support
  Python::Python
  GTest::gtest
  GTest::gtest_main
  Threads::Threads
)

target_include_directories(
  RuntimeTests
  PRIVATE
  ${PROJECT_SOURCE_DIR}/..
  ${GENERATED_HEADER_DIR}
  ${Python_INCLUDE_DIRS}
)

set_target_properties(
  RuntimeTests
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set(RUNTIME_TEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/runtime_test_root)

add_custom_command(
  TARGET RuntimeTests
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${RUNTIME_TEST_ROOT}
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/hir_tests
          ${RUNTIME_TEST_ROOT}/hir_tests
)
