#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# pyre-unsafe
# @noautodeps

import datetime
import glob
import os
import os.path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import textwrap
from enum import Enum
from functools import lru_cache
from typing import Callable

from setuptools import Extension, find_packages, setup
from setuptools.command.build import build as build
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

CHECKOUT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = "cinderx"
PYTHON_LIB_DIR = "cinderx/PythonLib"

MIN_GCC_VERSION = 13
GCC_PGO_AUDITED_TARGETS = {
    "_cinderx": os.path.join("CMakeFiles", "_cinderx.dir"),
    "borrowed": os.path.join("CMakeFiles", "borrowed.dir"),
    "cached-properties": os.path.join("CMakeFiles", "cached-properties.dir"),
    "common": os.path.join("CMakeFiles", "common.dir"),
    "immortalize": os.path.join("CMakeFiles", "immortalize.dir"),
    "interpreter": os.path.join("CMakeFiles", "interpreter.dir"),
    "jit": os.path.join("CMakeFiles", "jit.dir"),
    "parallel-gc": os.path.join("CMakeFiles", "parallel-gc.dir"),
    "static-python": os.path.join("CMakeFiles", "static-python.dir"),
}
GCC_PGO_REQUIRED_TARGETS = {
    "_cinderx": os.path.join("CMakeFiles", "_cinderx.dir"),
    "interpreter": os.path.join("CMakeFiles", "interpreter.dir"),
    "jit": os.path.join("CMakeFiles", "jit.dir"),
}
GCC_PGO_IGNORED_PROFILE_SUFFIXES = (
    os.path.join("cinderx", "Jit", "codegen", "arch", "unknown.cpp.gcda"),
    os.path.join("cinderx", "Jit", "codegen", "arch", "x86_64.cpp.gcda"),
    os.path.join("generated", "dummy-parallel-gc.c.gcda"),
)


def compute_package_version() -> str:
    """
    Compute a date-based version string.  Uses the UTC timezone for consistency.

    The returned format is YYYY.MM.DD.PP, where PP is a patch number.
    setuptools is going to normalize this anyway, so the two-character segments
    are going to get cut down if they have a leading zero.

    The patch number defaults to "00" but can be overridden with the
    CINDERX_VERSION_PATCH environment variable to allow multiple releases
    on the same day.

    When building from an extracted source distribution (sdist), the version
    is read from PKG-INFO to ensure consistency with the original release.
    """

    pkg_info_path = os.path.join(CHECKOUT_ROOT_DIR, "PKG-INFO")
    if os.path.exists(pkg_info_path):
        with open(pkg_info_path, "r") as f:
            for line in f:
                if line.startswith("Version:"):
                    version = line.split(":", 1)[1].strip()
                    print(f"Using version from PKG-INFO: {version}")
                    return version

    utc_now = datetime.datetime.now(datetime.timezone.utc)
    date_part = utc_now.strftime("%Y.%m.%d")
    patch = int(os.environ.get("CINDERX_VERSION_PATCH", "0"))
    return f"{date_part}.{patch:02}"


@lru_cache(maxsize=1)
def get_compiler() -> tuple[str, str]:
    """
    Prefers GCC if a new enough version is installed as this is what the
    cibuildwheel environment uses.

    Returns:
        A tuple of (c_compiler, cxx_compiler) paths.
    """
    cc_path = os.environ.get("CC")
    cxx_path = os.environ.get("CXX")
    if cc_path and cxx_path:
        print(f"Using CC and CXX from environment: {cc_path}, {cxx_path}")
        return (cc_path, cxx_path)

    gcc_path = shutil.which("gcc")
    gxx_path = shutil.which("g++")

    if gcc_path and gxx_path:
        try:
            result = subprocess.run(
                [gcc_path, "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            version_output = result.stdout

            # Parse GCC version from output like "gcc (GCC) 14.1.0"
            # The version is typically in the first line
            match = re.search(
                r"gcc.*?(\d+)\.(\d+)(?:\.(\d+))?", version_output, re.IGNORECASE
            )
            if match:
                major_version = int(match.group(1))
                print(f"Found GCC version {major_version}.{match.group(2)}")

                if major_version >= MIN_GCC_VERSION:
                    print(f"Using GCC: {gcc_path}, {gxx_path}")
                    return (gcc_path, gxx_path)
                else:
                    print(
                        f"GCC version {major_version} < {MIN_GCC_VERSION}, checking for Clang"
                    )
        except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
            print(f"Failed to determine GCC version: {e}, checking for Clang")

    # Fall back to Clang
    clang_path = shutil.which("clang")
    clangxx_path = shutil.which("clang++")

    if clang_path and clangxx_path:
        print(f"Using Clang: {clang_path}, {clangxx_path}")
        return (clang_path, clangxx_path)

    raise RuntimeError("Cannot find suitable C/C++ compiler (tried gcc and clang)")


class PgoStage(Enum):
    DISABLED = 0
    GENERATE = 1
    USE = 3


def compute_py_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def should_enable_adaptive_static_python(
    py_version: str,
    meta_python: bool,
    machine: str | None = None,
) -> bool:
    if meta_python and py_version == "3.12":
        return True

    if machine is None:
        machine = platform.machine()
    machine = machine.lower()

    return py_version == "3.14" and machine in {"aarch64", "arm64"}


def should_enable_lightweight_frames(
    py_version: str,
    meta_python: bool,
    machine: str | None = None,
) -> bool:
    if meta_python and py_version == "3.12":
        return True

    if machine is None:
        machine = platform.machine()
    machine = machine.lower()

    # Stage A rollout:
    # - keep existing meta 3.12 behavior
    # - enable by default for OSS 3.14 ARM only
    return py_version == "3.14" and machine in {"aarch64", "arm64"}


def is_env_flag_enabled(var: str, default: bool = False) -> bool:
    raw_value = os.environ.get(var)
    if raw_value is None:
        return default

    value = raw_value.strip().lower()
    if value in {"", "0", "false", "off", "no"}:
        return False
    if value in {"1", "true", "on", "yes"}:
        return True

    # Preserve historical behavior: any unknown non-empty value means enabled.
    return True


def validate_pgo_runtime(cinderx_module: object) -> None:
    import_error_getter = getattr(cinderx_module, "get_import_error", None)
    if not callable(import_error_getter):
        raise RuntimeError(
            "PGO workload validation failed: cinderx.get_import_error() is unavailable"
        )

    import_error = import_error_getter()
    if import_error is not None:
        raise RuntimeError(
            "PGO workload validation failed: _cinderx failed to import: "
            f"{import_error!r}"
        )

    is_initialized = getattr(cinderx_module, "is_initialized", None)
    if not callable(is_initialized):
        raise RuntimeError(
            "PGO workload validation failed: cinderx.is_initialized() is unavailable"
        )

    if not is_initialized():
        raise RuntimeError(
            "PGO workload validation failed: cinderx imported but _cinderx did not initialize"
        )


def build_pgo_workload_command() -> list[str]:
    workload_script = textwrap.dedent(
        """
        import sys

        import cinderx
        from setup import validate_pgo_runtime

        validate_pgo_runtime(cinderx)
        sys.argv.append("--pgo")

        def main():
            # This import must not be in the module body as it will start the tests
            # running, and those using multiprocessing will fail because the initial
            # process does not have "freeze support".
            import test.__main__

        if __name__ == "__main__":
            main()
        """
    )
    return [sys.executable, "-c", workload_script]


def collect_gcc_pgo_profile_counts(
    build_temp: str,
    required_targets: dict[str, str] | None = None,
) -> dict[str, int]:
    required_targets = required_targets or GCC_PGO_REQUIRED_TARGETS
    profile_counts = {}

    for target, relative_dir in required_targets.items():
        gcda_count = 0
        target_dir = os.path.join(build_temp, relative_dir)
        if os.path.isdir(target_dir):
            for root, _dirs, files in os.walk(target_dir):
                gcda_count += sum(1 for file_name in files if file_name.endswith(".gcda"))
        profile_counts[target] = gcda_count

    return profile_counts


def collect_gcc_pgo_missing_profile_files(
    build_temp: str,
    audited_targets: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    audited_targets = audited_targets or GCC_PGO_AUDITED_TARGETS
    missing_profiles = {}

    for target, relative_dir in audited_targets.items():
        missing_target_profiles = []
        target_dir = os.path.join(build_temp, relative_dir)
        if os.path.isdir(target_dir):
            for root, _dirs, files in os.walk(target_dir):
                existing_files = set(files)
                for file_name in files:
                    if not file_name.endswith(".o"):
                        continue
                    expected_gcda = file_name[:-2] + ".gcda"
                    if expected_gcda in existing_files:
                        continue
                    expected_gcda_path = os.path.join(root, expected_gcda)
                    normalized_path = os.path.normpath(expected_gcda_path)
                    if any(
                        normalized_path.endswith(os.path.normpath(suffix))
                        for suffix in GCC_PGO_IGNORED_PROFILE_SUFFIXES
                    ):
                        continue
                    missing_target_profiles.append(expected_gcda_path)
        if missing_target_profiles:
            missing_profiles[target] = sorted(missing_target_profiles)

    return missing_profiles


def require_gcc_pgo_profiles(
    build_temp: str,
    profile_counts: dict[str, int],
    required_targets: dict[str, str] | None = None,
) -> None:
    required_targets = required_targets or GCC_PGO_REQUIRED_TARGETS
    audit_failures = []
    missing_targets = []

    for target, relative_dir in required_targets.items():
        if profile_counts.get(target, 0) == 0:
            target_dir = os.path.join(build_temp, relative_dir)
            missing_targets.append(f"{target} ({target_dir})")

    if missing_targets:
        audit_failures.append(
            "missing .gcda files for required targets: " + ", ".join(missing_targets)
        )

    if audit_failures:
        raise RuntimeError("GCC PGO profile audit failed: " + " | ".join(audit_failures))


def run_pgo_workload(workload_cmd: list[str], workload_args: dict[str, object]) -> None:
    # CPython's --pgo test workload can fail intermittently; retry once to
    # avoid aborting an otherwise healthy instrumented build on a flaky run.
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(workload_cmd, **workload_args)
            return
        except subprocess.CalledProcessError as exc:
            if attempt == max_attempts:
                raise
            print(
                "PGO workload failed "
                f"(attempt {attempt}/{max_attempts}, exit={exc.returncode}); "
                "retrying..."
            )


class BuildCommand(build):
    # Don't use the setuptools default under "build/" as this clashes with
    # "build/fbcode_builder/" (auto-added to the OSS view of CinderX).
    def initialize_options(self):
        build.initialize_options(self)
        self.build_base = os.environ.get("CINDERX_BUILD_BASE", "scratch")

    def run(self) -> None:
        enable_pgo = is_env_flag_enabled("CINDERX_ENABLE_PGO")

        if enable_pgo:
            self._run_with_pgo()
        else:
            super().run()

    def _run_with_pgo(self) -> None:
        def print_section(title: str) -> None:
            separator = "=" * 70
            print(f"\n{separator}")
            print(title)
            print(separator)

        cc, _ = get_compiler()
        is_clang = "clang" in cc

        print_section("PGO STAGE 1/3: Building with profile generation instrumentation")

        stage1_build_ext_cmd = self.get_finalized_command("build_ext")
        stage1_build_ext_cmd.cinderx_pgo_stage = PgoStage.GENERATE

        # Run normal build process (BuildPy + BuildExt)
        super().run()

        print_section("PGO STAGE 2/3: Running profiling workload")

        workload_env = os.environ.copy()

        if is_clang:
            clang_pgo_dir = os.path.join(
                os.getcwd(), os.path.join(self.build_temp, "pgo_data")
            )
            os.makedirs(clang_pgo_dir, exist_ok=True)
            raw_profile_pattern = os.path.join(clang_pgo_dir, "code-%p.profraw")
            clang_merged_profile = os.path.join(clang_pgo_dir, "code.profdata")
            workload_env["LLVM_PROFILE_FILE"] = raw_profile_pattern

        # Add build output to PYTHONPATH so workload can import cinderx
        cinderx_so_dir = os.path.join(os.getcwd(), self.build_lib)
        pythonpath_entries = [cinderx_so_dir, CHECKOUT_ROOT_DIR]
        existing_pythonpath = workload_env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        workload_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

        # Uses the same default workload as CPython's PGO.
        workload_cmd = build_pgo_workload_command()

        print(f"Running workload with PYTHONPATH={workload_env['PYTHONPATH']}")
        workload_args = {
            "env": workload_env,
            "check": True,
        }
        if is_clang:
            workload_args["cwd"] = clang_pgo_dir
        run_pgo_workload(workload_cmd, workload_args)

        if is_clang:
            print_section("PGO STAGE 2b: Merging profile data")

            llvm_profdata = shutil.which("llvm-profdata")
            if not llvm_profdata:
                raise RuntimeError("Cannot find llvm-profdata")
            glob_path = os.path.join(clang_pgo_dir, "*.profraw")
            profraw_files = glob.glob(os.path.join(clang_pgo_dir, "*.profraw"))

            if not profraw_files:
                raise RuntimeError(
                    f"No profile data generated when searching {glob_path}"
                )

            print(f"Found {len(profraw_files)} profile files to merge")
            merge_cmd = [
                llvm_profdata,
                "merge",
                "-output=" + clang_merged_profile,
            ] + profraw_files
            subprocess.run(merge_cmd, check=True)
            print(f"Merged profile written to {clang_merged_profile}")
        else:
            print_section("PGO STAGE 2b: Auditing GCC profile data")
            build_temp_dir = os.path.join(os.getcwd(), self.build_temp)
            profile_counts = collect_gcc_pgo_profile_counts(
                build_temp_dir,
                GCC_PGO_AUDITED_TARGETS,
            )
            missing_profile_files = collect_gcc_pgo_missing_profile_files(
                build_temp_dir,
                GCC_PGO_AUDITED_TARGETS,
            )
            for target, count in profile_counts.items():
                missing_count = len(missing_profile_files.get(target, []))
                print(
                    f"  {target}: found {count} .gcda files, "
                    f"missing {missing_count} expected profiles"
                )
            if missing_profile_files:
                missing_summary = ", ".join(
                    f"{target}={len(file_paths)}"
                    for target, file_paths in sorted(missing_profile_files.items())
                )
                print(
                    "  Note: some per-object profiles are missing "
                    f"({missing_summary}); GCC will use partial-training PGO."
                )
            require_gcc_pgo_profiles(build_temp_dir, profile_counts)

        print_section("PGO STAGE 3/3: Rebuilding with profile-guided optimizations")

        print("Cleaning build artifacts (keep profile data + python libs)...")

        # IMPORTANT: We need to preserve CMakeCache.txt to avoid re-running
        # compiler feature detection. For some reason asmjit seems to
        # consistently change its mind on the need for -fmerge-all-constants
        # after the instrumentation build, and this invalidates some profiling
        # data. However, we DO need to force CMake to reconfigure to pick up
        # the new PGO_STAGE=USE environment variable. We do this by removing
        # the CMakeFiles directory but keeping CMakeCache.txt.
        #
        # For GCC PGO, we must preserve .gcda files (profile data generated
        # during the workload run) as they are stored alongside object files
        # in CMakeFiles/. We selectively remove only what we need to force
        # reconfiguration.

        cmake_files = os.path.join(self.build_temp, "CMakeFiles")
        if os.path.exists(cmake_files):
            print(f"  Cleaning {cmake_files} (preserving .gcda/.gcno for GCC PGO)")
            for root, _dirs, files in os.walk(cmake_files, topdown=False):
                for f in files:
                    file_path = os.path.join(root, f)
                    # Preserve GCC PGO profiling files
                    if f.endswith((".gcda", ".gcno")):
                        continue
                    os.remove(file_path)
                # Remove empty directories, but only if they don't contain
                # preserved files (the topdown=False walk handles this)
                try:
                    os.rmdir(root)  # Only removes if empty
                except OSError:
                    pass  # Directory not empty, contains .gcda/.gcno files

        # Remove all object files and libraries to force rebuild
        for rm_root in (self.build_temp, self.build_lib):
            if os.path.exists(rm_root):
                for root, _dirs, files in os.walk(rm_root):
                    # Skip the pgo_data directory
                    if "pgo_data" in root:
                        continue
                    for f in files:
                        if f.endswith((".o", ".so", ".a")):
                            file_path = os.path.join(root, f)
                            print(f"  Removing {file_path}")
                            os.remove(file_path)

        print("Running rebuild with PGO optimizations...")

        # Create a new build_ext command instance with extensions list. The old
        # one from Stage 1 consumed its inputs and cannot be reused.
        stage3_build_ext_cmd = BuildExt(self.distribution)
        stage3_build_ext_cmd.build_lib = self.build_lib
        stage3_build_ext_cmd.build_temp = self.build_temp
        stage3_build_ext_cmd.inplace = False
        stage3_build_ext_cmd.force = True
        stage3_build_ext_cmd.cinderx_pgo_stage = PgoStage.USE
        if is_clang:
            stage3_build_ext_cmd.cinderx_pgo_profile_path = clang_merged_profile
        stage3_build_ext_cmd.finalize_options()
        stage3_build_ext_cmd.run()

        print_section("PGO BUILD COMPLETE!")


class BuildPy(build_py):
    def run(self) -> None:
        # I have no idea what is supposed to set this up, but if it isn't set
        # we'll get an AttributeError at runtime.
        if not hasattr(self.distribution, "namespace_packages"):
            self.distribution.namespace_packages = None

        super().run()

        # Copy opcodes/${PY_VERSION}/opcode.py to cinderx/opcode.py.
        py_version = compute_py_version()
        opcode_src = os.path.join(PYTHON_LIB_DIR, f"opcodes/{py_version}/opcode.py")
        out_path = self.get_module_outfile(self.build_lib, ["cinderx"], "opcode")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.copy_file(
            opcode_src,
            out_path,
            preserve_mode=False,
        )

        # Editable installs extend sys.path with the source tree rather than the
        # temporary build output, so keep the generated opcode module alongside
        # the package sources as well.
        source_opcode_path = os.path.join(self.get_package_dir("cinderx"), "opcode.py")
        self.copy_file(
            opcode_src,
            source_opcode_path,
            preserve_mode=False,
        )

        # For OSS builds always surface the CinderX import errors
        dev_build_file = os.path.join(self.get_package_dir("cinderx"), ".dev_build")
        print(f"Writing .dev_build file to {dev_build_file}")
        with open(dev_build_file, "w") as f:
            f.write("\n")
        # Copy .pth file to build_lib root for auto-import on Python startup
        pth_source = os.path.join(PYTHON_LIB_DIR, "cinderx.pth")
        pth_dest = os.path.join(self.build_lib, "cinderx.pth")
        print(f"Copying .pth file to {pth_dest}")
        self.copy_file(pth_source, pth_dest, preserve_mode=False)

class CMakeExtension(Extension):
    """
    Subclass to indicate to BuildExt that this should be handled specially.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name, sources=[])


class BuildExt(build_ext):
    def __init__(self, distribution: Distribution) -> None:
        super().__init__(distribution)
        self.cinderx_pgo_stage = PgoStage.DISABLED
        self.cinderx_pgo_profile_path: str | None = None

    def run(self) -> None:
        # Partition into CMake extensions and everything else.
        cmake_extensions = []
        other_extensions = []
        # pyre-ignore[16]: No pyre types for build_ext.
        for extension in self.extensions:
            if isinstance(extension, CMakeExtension):
                cmake_extensions.append(extension)
            else:
                other_extensions.append(extension)

        # Handle CMake specially, leave everything else to super().
        for extension in cmake_extensions:
            self._run_cmake(extension)

        # pyre-ignore[16]: No pyre types for build_ext.
        self.extensions = other_extensions
        super().run()

    def _run_cmake(self, extension: CMakeExtension) -> None:
        # pyre-ignore[16]: No pyre types for build_ext.
        build_dir = self.build_temp
        os.makedirs(build_dir, exist_ok=True)

        # pyre-ignore[16]: No pyre types for build_ext.
        extension_dir = os.path.abspath(self.get_ext_fullpath(extension.name))
        os.makedirs(extension_dir, exist_ok=True)

        cc, cxx = get_compiler()

        build_type = os.environ.get("CMAKE_BUILD_TYPE", "RelWithDebInfo")
        verbose_makefile = os.environ.get("CMAKE_VERBOSE_MAKEFILE", "OFF")
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.dirname(extension_dir)}",
            f"-DCMAKE_C_COMPILER={cc}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
            f"-DCMAKE_VERBOSE_MAKEFILE:BOOL={verbose_makefile}",
        ]

        if self.cinderx_pgo_stage == PgoStage.GENERATE:
            cmake_args.append("-DENABLE_PGO_GENERATE=ON")
            cmake_args.append("-DENABLE_PGO_USE=OFF")
        elif self.cinderx_pgo_stage == PgoStage.USE:
            cmake_args.append("-DENABLE_PGO_GENERATE=OFF")
            cmake_args.append("-DENABLE_PGO_USE=ON")
            if self.cinderx_pgo_profile_path:
                cmake_args.append(f"-DPGO_PROFILE_FILE={self.cinderx_pgo_profile_path}")

        # LTO configuration
        enable_lto = is_env_flag_enabled("CINDERX_ENABLE_LTO")
        if enable_lto:
            cmake_args.append("-DENABLE_LTO=ON")
            print("Building with LTO enabled (full LTO)")
        else:
            cmake_args.append("-DENABLE_LTO=OFF")

        options: dict[str, str] = {}

        def set_option(var: str, default: object) -> None:
            if type(default) == bool:
                default = int(default)
            if type(default) == int:
                default = str(default)
            if type(default) != str:
                raise ValueError(f"Not sure what to do with default value {default}")

            value = os.environ.get(var, default)
            options[var] = value

        # Python version is always the same as what's running setuptools.
        py_version = compute_py_version()
        options["PY_VERSION"] = py_version
        options["Python_ROOT_DIR"] = self._find_python()

        meta_python = "+meta" in sys.version
        linux = sys.platform == "linux"
        meta_312 = meta_python and py_version == "3.12"
        is_314plus = py_version == "3.14" or py_version == "3.15"

        set_option("META_PYTHON", meta_python)
        enable_static_python = is_env_flag_enabled("ENABLE_STATIC_PYTHON", True)
        set_option("ENABLE_STATIC_PYTHON", enable_static_python)
        set_option(
            "ENABLE_ADAPTIVE_STATIC_PYTHON",
            enable_static_python
            and should_enable_adaptive_static_python(py_version, meta_python),
        )
        if not enable_static_python:
            options["ENABLE_ADAPTIVE_STATIC_PYTHON"] = "0"
        set_option("ENABLE_DISASSEMBLER", False)
        set_option("ENABLE_ELF_READER", linux)
        set_option("ENABLE_EVAL_HOOK", meta_312)
        set_option("ENABLE_FUNC_EVENT_MODIFY_QUALNAME", meta_312)
        set_option("ENABLE_GENERATOR_AWAITER", meta_312)
        set_option("ENABLE_INTERPRETER_LOOP", meta_312 or is_314plus)
        set_option("ENABLE_LAZY_IMPORTS", meta_312)
        set_option(
            "ENABLE_LIGHTWEIGHT_FRAMES",
            should_enable_lightweight_frames(py_version, meta_python),
        )
        set_option("ENABLE_PARALLEL_GC", meta_312)
        set_option("ENABLE_PEP523_HOOK", meta_312 or is_314plus)
        set_option("ENABLE_PERF_TRAMPOLINE", meta_312)
        set_option("ENABLE_SYMBOLIZER", linux)
        set_option("ENABLE_USDT", linux)

        for name, value in options.items():
            cmake_args.append(f"-D{name}={value}")

        build_jobs = int(os.environ.get("CINDERX_BUILD_JOBS", os.cpu_count() or 1))

        build_args = [
            "--config",
            build_type,
            "--",
            "-j",
            str(build_jobs),
        ]

        # pyre-ignore[16]: No pyre types for build_ext.
        self.spawn(["cmake"] + cmake_args + ["-B", build_dir, CHECKOUT_ROOT_DIR])
        self.spawn(["cmake", "--build", build_dir] + build_args)

    def _find_python(self) -> str:
        override = os.environ.get("Python_ROOT_DIR")
        if override:
            return override
        # Normally this would use "data", but that goes to a temporary build directory
        # under uv.  Work off of the include directory instead.
        include_dir = sysconfig.get_path("include")
        return os.path.join(include_dir, "..", "..")


def main() -> None:
    setup(
        name="cinderx",
        version=compute_package_version(),
        ext_modules=[
            CMakeExtension(name="_cinderx"),
        ],
        cmdclass={
            "build": BuildCommand,
            "build_py": BuildPy,
            "build_ext": BuildExt,
        },
        packages=find_packages(where=PYTHON_LIB_DIR, exclude=["test_cinderx*"]),
        package_dir={"": PYTHON_LIB_DIR},
        package_data={"cinderx": [".dev_build"]},
    )


if __name__ == "__main__":
    main()
