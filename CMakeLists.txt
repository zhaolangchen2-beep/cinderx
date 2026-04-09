# (c) Meta Platforms, Inc. and affiliates.

message("=====================================================================")
message("=                            WARNING                                =")
message("=====================================================================")
message("=                                                                   =")
message("= This CMake build is not well supported, and is primarily used for =")
message("= experimentation.  Do not expect this to work on your machine.     =")
message("=====================================================================")
message("")

cmake_minimum_required(VERSION 3.12)
project(_cinderx)

set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/cinderx)
set(CINDERX_SOURCE_DIR ${PROJECT_SOURCE_DIR})

##############################################################################
# PGO (Profile-Guided Optimization) configuration

option(ENABLE_PGO_GENERATE "Build with PGO profile generation" OFF)
option(ENABLE_PGO_USE "Build with PGO profile use" OFF)
set(PGO_PROFILE_FILE "" CACHE STRING "Path to PGO profile data")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(MACOS 1)
else()
  set(MACOS 0)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(USING_CLANG TRUE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(USING_GCC TRUE)
endif()

##############################################################################
# Figure out capabilities / supported features and set compiler flags
# appropriately.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SHARED_FLAGS "-DPy_BUILD_CORE -DPy_BUILD_CORE_MODULE -fPIC -Wall -Wextra -Wno-attributes -Wno-c99-designator -Wno-c++11-narrowing -Wno-cast-function-type -Wno-cast-function-type-mismatch -Wno-deprecated-declarations -Wno-missing-field-initializers -Wno-null-pointer-subtraction -Wno-parentheses -Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-parameter")

macro(set_flag VAR)
  if(DEFINED ${VAR})
    if(${VAR})
      set(SHARED_FLAGS "-D${VAR} ${SHARED_FLAGS}")
    endif()
  endif()
endmacro()

set_flag(ENABLE_ADAPTIVE_STATIC_PYTHON)
set_flag(ENABLE_DISASSEMBLER)
set_flag(ENABLE_ELF_READER)
set_flag(ENABLE_EVAL_HOOK)
set_flag(ENABLE_FUNC_EVENT_MODIFY_QUALNAME)
set_flag(ENABLE_GENERATOR_AWAITER)
set_flag(ENABLE_INTERPRETER_LOOP)
set_flag(ENABLE_LAZY_IMPORTS)
set_flag(ENABLE_LIGHTWEIGHT_FRAMES)
set_flag(ENABLE_PARALLEL_GC)
set_flag(ENABLE_PEP523_HOOK)
set_flag(ENABLE_PERF_TRAMPOLINE)
set_flag(ENABLE_SYMBOLIZER)
set_flag(ENABLE_USDT)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SHARED_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SHARED_FLAGS}")

##############################################################################
# Apply LTO (Link-Time Optimization) if enabled

option(ENABLE_LTO "Enable Link-Time Optimization (full)" OFF)
set(CINDERX_FIRST_PARTY_COMPILE_OPTIONS "")
set(CINDERX_FIRST_PARTY_LINK_OPTIONS "")

if(ENABLE_LTO)
  # Only support Linux
  if(MACOS)
    message(FATAL_ERROR "LTO is only supported on Linux")
  endif()

  message(STATUS "LTO: Enabled (full LTO)")

  if(NOT USING_CLANG AND NOT USING_GCC)
    message(FATAL_ERROR "LTO is only supported with Clang or GCC compilers")
  endif()

  # For Clang, we need llvm-ar instead of regular ar
  if(USING_CLANG)
    find_program(LLVM_AR llvm-ar)
    if(NOT LLVM_AR)
      message(FATAL_ERROR "llvm-ar is required for LTO with Clang but was not found")
    endif()
    set(CMAKE_AR ${LLVM_AR})
    message(STATUS "LTO: Using llvm-ar: ${CMAKE_AR}")

    list(APPEND CINDERX_FIRST_PARTY_COMPILE_OPTIONS -flto)
    list(APPEND CINDERX_FIRST_PARTY_LINK_OPTIONS -flto)
  elseif(USING_GCC)
    list(APPEND CINDERX_FIRST_PARTY_COMPILE_OPTIONS -flto -ffat-lto-objects)
    list(APPEND CINDERX_FIRST_PARTY_LINK_OPTIONS -flto -fuse-linker-plugin)
  endif()
endif()

##############################################################################
# Apply PGO flags if enabled

if(ENABLE_PGO_GENERATE)
  message(STATUS "PGO: Building with profile generation instrumentation")
  if(USING_CLANG)
    list(APPEND CINDERX_FIRST_PARTY_COMPILE_OPTIONS -fprofile-instr-generate)
    list(APPEND CINDERX_FIRST_PARTY_LINK_OPTIONS -fprofile-instr-generate)
  elseif(USING_GCC)
    list(APPEND CINDERX_FIRST_PARTY_COMPILE_OPTIONS -fprofile-generate)
    list(APPEND CINDERX_FIRST_PARTY_LINK_OPTIONS -fprofile-generate)
  else()
    message(WARNING "PGO profile generation is not supported with compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()
endif()

if(ENABLE_PGO_USE)
  message(STATUS "PGO: Building with profile-guided optimizations")
  if(USING_CLANG)
    if(NOT PGO_PROFILE_FILE)
      message(FATAL_ERROR "PGO_PROFILE_FILE must be set when ENABLE_PGO_USE=ON")
    endif()
    if(NOT EXISTS "${PGO_PROFILE_FILE}")
      message(FATAL_ERROR "PGO profile file does not exist: ${PGO_PROFILE_FILE}")
    endif()
    message(STATUS "PGO: Using profile data from ${PGO_PROFILE_FILE}")
    list(APPEND CINDERX_FIRST_PARTY_COMPILE_OPTIONS -fprofile-instr-use=${PGO_PROFILE_FILE})
  elseif(USING_GCC)
    message(STATUS "PGO: Using GCC profile data from build directory")
    list(
      APPEND
      CINDERX_FIRST_PARTY_COMPILE_OPTIONS
      -fprofile-use
      -fprofile-correction
      -fprofile-partial-training
      -Wno-missing-profile
    )
  else()
    message(WARNING "PGO profile use is not supported with compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()
endif()

function(cinderx_apply_first_party_optimizations target_name)
  if(CINDERX_FIRST_PARTY_COMPILE_OPTIONS)
    target_compile_options(
      ${target_name}
      PRIVATE
      ${CINDERX_FIRST_PARTY_COMPILE_OPTIONS}
    )
  endif()

  if(CINDERX_FIRST_PARTY_LINK_OPTIONS)
    get_target_property(target_type ${target_name} TYPE)
    if(
      target_type STREQUAL "SHARED_LIBRARY"
      OR target_type STREQUAL "MODULE_LIBRARY"
      OR target_type STREQUAL "EXECUTABLE"
    )
      target_link_options(
        ${target_name}
        PRIVATE
        ${CINDERX_FIRST_PARTY_LINK_OPTIONS}
      )
    endif()
  endif()
endfunction()

set(GENERATED_HEADER_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_HEADER_DIR})

##############################################################################
# Dependencies

include(FetchContent)

########################################
# python

find_package(
  Python ${PY_VERSION} EXACT
  COMPONENTS Interpreter Development.Module
  REQUIRED)

# Some of our files are partially generated from CPython source, and they
# expect to be able to include files relative to Include/internal.

########################################
# asmjit

set(ASMJIT_STATIC TRUE)

FetchContent_Declare(
  asmjit
  URL https://codeload.github.com/asmjit/asmjit/tar.gz/cecc73f2979e9704c81a2c2ec79a7475b31c56ac
  URL_HASH SHA256=8f18fd88c384c206e3a3da4160e6d79320ac3c5c19d956858ec18f9d57ccf02d
)
FetchContent_MakeAvailable(asmjit)

########################################
# fmt

FetchContent_Declare(
  fmt
  URL https://codeload.github.com/fmtlib/fmt/tar.gz/11.2.0
  URL_HASH SHA256=bc23066d87ab3168f27cef3e97d545fa63314f5c79df5ea444d41d56f962c6af
)
FetchContent_MakeAvailable(fmt)

########################################
# parallel-hashmap

FetchContent_Declare(
  parallel-hashmap
  URL https://codeload.github.com/greg7mdp/parallel-hashmap/tar.gz/896f1a03e429c45d9fe9638e892fc1da73befadd
  URL_HASH SHA256=9a44071db1760ca18b56815ba719b61651d4b32efbd9f13e8e41fe9a5084f65d
)
FetchContent_MakeAvailable(parallel-hashmap)

########################################
# usdt

FetchContent_Declare(
  usdt
  URL https://codeload.github.com/libbpf/usdt/tar.gz/f4ea2f524efa80d062f4d586d78daafb83dc7d24
  URL_HASH SHA256=69467234c3c009952fea99dd057e5200160603c7e3e04ecb74458e29746e5b95
)
FetchContent_MakeAvailable(usdt)

# Get directory format for `#include <usdt/usdt.h>` to work.
set(USDT_DIR ${GENERATED_HEADER_DIR}/usdt)
file(MAKE_DIRECTORY ${USDT_DIR})
file(COPY ${usdt_SOURCE_DIR}/usdt.h DESTINATION ${USDT_DIR})

########################################
# zlib

find_package(ZLIB)
if(NOT ZLIB_FOUND)
  message(STATUS "ZLIB not found locally; fetching zlib from GitHub archive")
  set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    zlib
    URL https://codeload.github.com/madler/zlib/tar.gz/v1.3.1
    URL_HASH SHA256=17e88863f3600672ab49182f217281b6fc4d3c762bde361935e436a95214d05c
  )
  FetchContent_MakeAvailable(zlib)
  if(TARGET zlibstatic AND NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
  elseif(TARGET zlib AND NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB ALIAS zlib)
  endif()
endif()

option(BUILD_RUNTIME_TESTS "Build the RuntimeTests gtest binary" OFF)

if(BUILD_RUNTIME_TESTS)
  FetchContent_Declare(
    googletest
    URL https://codeload.github.com/google/googletest/tar.gz/v1.14.0
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  find_package(Python ${PY_VERSION} EXACT COMPONENTS Development.Embed REQUIRED)
endif()
list(GET Python_INCLUDE_DIRS 0 PYTHON_PRIMARY_INCLUDE_DIR)
list(APPEND Python_INCLUDE_DIRS "${PYTHON_PRIMARY_INCLUDE_DIR}/internal")

##############################################################################
# Build cinderx

include_directories(
  ${PROJECT_SOURCE_DIR}/..
  ${GENERATED_HEADER_DIR}
  ${Python_INCLUDE_DIRS}
  ${parallel-hashmap_SOURCE_DIR}
  ${usdt_SOURCE_DIR}/..
)

########################################
# Common/

file(GLOB_RECURSE COMMON_SOURCES ${PROJECT_SOURCE_DIR}/Common/*.cpp)
add_library(common ${COMMON_SOURCES})
cinderx_apply_first_party_optimizations(common)
target_link_libraries(common PRIVATE asmjit::asmjit fmt::fmt ZLIB::ZLIB)

########################################
# UpstreamBorrow/

if (${PY_VERSION} EQUAL 3.12 OR ${PY_VERSION} EQUAL 3.14 OR ${PY_VERSION} EQUAL 3.15)
  set(BORROWED_C ${PROJECT_SOURCE_DIR}/UpstreamBorrow/borrowed-${PY_VERSION}.gen_cached.c)
else()
  set(BORROWED_C "${GENERATED_HEADER_DIR}/dummy-borrowed.c")
  file(WRITE ${BORROWED_C} "")
endif()

add_library(borrowed ${BORROWED_C})
cinderx_apply_first_party_optimizations(borrowed)

########################################
# CachedProperties/

file(GLOB_RECURSE CACHED_PROPERTY_SOURCES ${PROJECT_SOURCE_DIR}/CachedProperties/*.c)
add_library(cached-properties ${CACHED_PROPERTY_SOURCES})
cinderx_apply_first_party_optimizations(cached-properties)
target_link_libraries(cached-properties PRIVATE common borrowed)

########################################
# Interpreter/

set(INTERP_SOURCES
  ${PROJECT_SOURCE_DIR}/Interpreter/cinder_opcode.c
  ${PROJECT_SOURCE_DIR}/Interpreter/interpreter_base.cpp
  ${PROJECT_SOURCE_DIR}/Interpreter/iter_helpers.c
)

if (${PY_VERSION} EQUAL 3.12 OR ${PY_VERSION} EQUAL 3.14 OR ${PY_VERSION} EQUAL 3.15)
  set(SOURCE_CINDERX_OPCODE_TARGETS_H ${PROJECT_SOURCE_DIR}/Interpreter/${PY_VERSION}/cinderx_opcode_targets.h)
  set(SOURCE_CINDER_OPCODE_H ${PROJECT_SOURCE_DIR}/Interpreter/${PY_VERSION}/cinder_opcode.h)
  set(SOURCE_CINDER_OPCODE_IDS_H ${PROJECT_SOURCE_DIR}/Interpreter/${PY_VERSION}/cinder_opcode_ids.h)
  set(SOURCE_CINDER_OPCODE_METADATA_H ${PROJECT_SOURCE_DIR}/Interpreter/${PY_VERSION}/cinder_opcode_metadata.h)

  if (${ENABLE_INTERPRETER_LOOP})
    set(INTERP_SOURCES ${INTERP_SOURCES} ${PROJECT_SOURCE_DIR}/Interpreter/${PY_VERSION}/interpreter.c)
  endif()
endif()

# Generate cinderx/Interpreter header files.

set(CINDERX_OPCODE_TARGETS_H ${GENERATED_HEADER_DIR}/cinderx/Interpreter/cinderx_opcode_targets.h)
set(CINDER_OPCODE_H ${GENERATED_HEADER_DIR}/cinderx/Interpreter/cinder_opcode.h)
set(CINDER_OPCODE_IDS_H ${GENERATED_HEADER_DIR}/cinderx/Interpreter/cinder_opcode_ids.h)
set(CINDER_OPCODE_METADATA_H ${GENERATED_HEADER_DIR}/cinderx/Interpreter/cinder_opcode_metadata.h)

file(READ ${SOURCE_CINDERX_OPCODE_TARGETS_H} FILE_CONTENTS)
file(WRITE ${CINDERX_OPCODE_TARGETS_H} "${FILE_CONTENTS}")

file(READ ${SOURCE_CINDER_OPCODE_IDS_H} FILE_CONTENTS)
file(WRITE ${CINDER_OPCODE_IDS_H} "${FILE_CONTENTS}")

file(READ ${SOURCE_CINDER_OPCODE_METADATA_H} FILE_CONTENTS)
file(WRITE ${CINDER_OPCODE_METADATA_H} "${FILE_CONTENTS}")

file(READ ${SOURCE_CINDER_OPCODE_H} FILE_CONTENTS)
file(WRITE ${CINDER_OPCODE_H} "${FILE_CONTENTS}")

add_library(interpreter ${INTERP_SOURCES})
cinderx_apply_first_party_optimizations(interpreter)
target_include_directories(
  interpreter
  PUBLIC
  ${PROJECT_SOURCE_DIR}/Interpreter/${PY_VERSION}/Includes
)

########################################
# Immortalize/

file(GLOB_RECURSE IMMORTALIZE_SOURCES ${PROJECT_SOURCE_DIR}/Immortalize/*.cpp)
add_library(immortalize ${IMMORTALIZE_SOURCES})
cinderx_apply_first_party_optimizations(immortalize)
target_link_libraries(immortalize PRIVATE asmjit::asmjit fmt::fmt)

########################################
# StaticPython/

file(GLOB_RECURSE STATIC_PYTHON_SOURCES ${PROJECT_SOURCE_DIR}/StaticPython/*.c)
add_library(static-python ${STATIC_PYTHON_SOURCES})
cinderx_apply_first_party_optimizations(static-python)
target_link_libraries(static-python PRIVATE fmt::fmt asmjit::asmjit common borrowed)

########################################
# Jit/

file(GLOB_RECURSE JIT_SOURCES ${PROJECT_SOURCE_DIR}/Jit/*.cpp ${PROJECT_SOURCE_DIR}/Jit/*.c)
list(FILTER JIT_SOURCES EXCLUDE REGEX ".*\\.gen_cached\\.c")
if (${PY_VERSION} EQUAL 3.12 OR ${PY_VERSION} EQUAL 3.14 OR ${PY_VERSION} EQUAL 3.15)
  list(APPEND JIT_SOURCES "${PROJECT_SOURCE_DIR}/Jit/generators_borrowed_${PY_VERSION}.gen_cached.c")
endif()

add_library(jit ${JIT_SOURCES})
cinderx_apply_first_party_optimizations(jit)
target_link_libraries(jit PRIVATE asmjit::asmjit interpreter fmt::fmt)

########################################
# ParallelGC/

if (${ENABLE_PARALLEL_GC})
  file(GLOB_RECURSE PARALLEL_GC_SOURCES ${PROJECT_SOURCE_DIR}/ParallelGC/*.c)
else()
  set(PARALLEL_GC_SOURCES "${GENERATED_HEADER_DIR}/dummy-parallel-gc.c")
  file(WRITE ${PARALLEL_GC_SOURCES} "")
endif()
add_library(parallel-gc ${PARALLEL_GC_SOURCES})
cinderx_apply_first_party_optimizations(parallel-gc)

########################################
# _cinderx.cpp

set(SOURCES
  ${PROJECT_SOURCE_DIR}/_cinderx.cpp
  ${PROJECT_SOURCE_DIR}/_cinderx-lib.cpp
  ${PROJECT_SOURCE_DIR}/async_lazy_value.cpp
  ${PROJECT_SOURCE_DIR}/module_state.cpp
  ${PROJECT_SOURCE_DIR}/module_c_state.cpp
  ${PROJECT_SOURCE_DIR}/python_runtime.cpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
cinderx_apply_first_party_optimizations(${PROJECT_NAME})

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
  Python::Module
  borrowed cached-properties common immortalize interpreter jit parallel-gc static-python
  asmjit::asmjit fmt::fmt)

# macOS doesn't allow depending on other shared libraries by default.
if (${MACOS})
  target_link_options(${PROJECT_NAME} PRIVATE -undefined dynamic_lookup)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX "")
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "_cinderx.so")

if(BUILD_RUNTIME_TESTS)
  add_subdirectory(${PROJECT_SOURCE_DIR}/RuntimeTests)
endif()
