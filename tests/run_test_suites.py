#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import fnmatch
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
TESTSCRIPTS_RUNNER = REPO_ROOT / "cinderx" / "TestScripts" / "cinder_test_runner312.py"
DEFAULT_PYTHON = "python3"
DEFAULT_GCC_ROOT = Path("/opt/gcc-14")
DEFAULT_PY_VERSION = "3.14"
RUNTIME_BLACKLIST = REPO_ROOT / "tests" / "runtime_blacklist.txt"


@dataclass
class CaseResult:
    name: str
    status: str
    rc: int
    log: str
    note: str = ""


@dataclass
class SuiteRunSummary:
    name: str
    total: int
    counts: dict[str, int]
    output_dir: str
    artifacts: list[str]


@dataclass
class CoverageOverview:
    name: str
    covered: int
    total: int
    percent: float


def load_compat_module():
    compat_path = REPO_ROOT / "cinderx" / "PythonLib" / "cinderx" / "_compat.py"
    spec = importlib.util.spec_from_file_location("cinderx_test_runner_compat", compat_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load compatibility policy from {compat_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def is_oss_feature_enabled(py_version: str, machine: str, feature: str) -> bool:
    return bool(load_compat_module().is_oss_feature_enabled(py_version, machine, feature))


def compute_cmake_feature_options(py_version: str) -> dict[str, str]:
    machine = platform.machine()
    is_314plus = py_version in {"3.14", "3.15"}
    return {
        "ENABLE_ADAPTIVE_STATIC_PYTHON": (
            "ON" if is_oss_feature_enabled(py_version, machine, "adaptive_static_python") else "OFF"
        ),
        "ENABLE_EVAL_HOOK": "OFF",
        "ENABLE_INTERPRETER_LOOP": "ON" if is_314plus else "OFF",
        "ENABLE_LIGHTWEIGHT_FRAMES": (
            "ON" if is_oss_feature_enabled(py_version, machine, "lightweight_frames") else "OFF"
        ),
        "ENABLE_PEP523_HOOK": "ON" if is_314plus else "OFF",
    }


def default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return WORKSPACE_ROOT / "cov" / "ut" / stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for PythonLib and RuntimeTests."
    )
    parser.add_argument(
        "-t",
        "--target",
        choices=("all", "pythonlib", "runtime"),
        default="all",
        help="Target test family to run.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output_root(),
        help="Output root directory.",
    )
    parser.add_argument(
        "-f",
        "--filter",
        action="append",
        default=[],
        help="Filter pattern. PythonLib uses module names; runtime uses gtest filter.",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List tests only, do not execute.",
    )
    parser.add_argument(
        "-c",
        "--coverage",
        action="store_true",
        help="Enable coverage collection.",
    )
    parser.add_argument(
        "--python-exe",
        default=DEFAULT_PYTHON,
        help="Python executable for PythonLib tests.",
    )
    parser.add_argument(
        "--runtime-build-dir",
        type=Path,
        help="Override runtime build directory.",
    )
    parser.add_argument(
        "--runtime-binary",
        type=Path,
        help="Override RuntimeTests binary path.",
    )
    parser.add_argument(
        "--runtime-cwd",
        type=Path,
        help="Override RuntimeTests working directory.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        default=True,
        help="Continue after failures or crashes. Enabled by default.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip building RuntimeTests.",
    )
    parser.add_argument(
        "--gcc-root",
        type=Path,
        default=DEFAULT_GCC_ROOT,
        help="GCC14 toolset root for runtime build/coverage.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_case_results(results: list[CaseResult], output_dir: Path, name: str) -> None:
    tsv_path = output_dir / "summary.tsv"
    md_path = output_dir / "summary.md"
    json_path = output_dir / "tests.json"

    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("test\tstatus\trc\tlog\tnote\n")
        for result in results:
            f.write(
                f"{result.name}\t{result.status}\t{result.rc}\t{result.log}\t{result.note}\n"
            )

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {name} Summary\n\n")
        f.write("| test | status | rc | note | log |\n")
        f.write("| --- | --- | ---: | --- | --- |\n")
        for result in results:
            note = result.note.replace("\n", " ").strip()
            f.write(
                f"| `{result.name}` | `{result.status}` | `{result.rc}` | {note} | `{result.log}` |\n"
            )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(result) for result in results], f, indent=2)


def summarize_case_results(results: list[CaseResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return dict(sorted(counts.items()))


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def format_start_banner(args: argparse.Namespace) -> str:
    runtime_build_dir = pick_runtime_build_dir(args)
    runtime_binary = pick_runtime_binary(args, runtime_build_dir)
    runtime_cwd = pick_runtime_cwd(args, runtime_binary)
    lines = [
        "=" * 72,
        "CinderX UT task starting",
        "=" * 72,
        f"target             : {args.target}",
        f"output             : {args.output.resolve()}",
        f"filter             : {args.filter if args.filter else ['<none>']}",
        f"list_only          : {args.list}",
        f"coverage           : {args.coverage}",
        f"python_exe         : {args.python_exe}",
        f"runtime_build_dir  : {runtime_build_dir}",
        f"runtime_binary     : {runtime_binary}",
        f"runtime_cwd        : {runtime_cwd}",
        f"keep_going         : {args.keep_going}",
        f"no_build           : {args.no_build}",
        f"gcc_root           : {args.gcc_root}",
        "=" * 72,
    ]
    return "\n".join(lines)


def format_finish_summary(
    target: str,
    output_root: Path,
    summaries: list[SuiteRunSummary],
    coverage_overviews: list[CoverageOverview] | None = None,
) -> str:
    lines = [
        "=" * 72,
        "CinderX UT task finished",
        "=" * 72,
        f"target             : {target}",
        f"output_root        : {output_root}",
    ]
    if not summaries:
        lines.append("suites             : none")
    for summary in summaries:
        lines.extend(
            [
                f"{summary.name}_total      : {summary.total}",
                f"{summary.name}_counts     : {format_counts(summary.counts)}",
                f"{summary.name}_output     : {summary.output_dir}",
            ]
        )
        if summary.artifacts:
            lines.append(
                f"{summary.name}_artifacts  : {', '.join(summary.artifacts)}"
            )
    if coverage_overviews:
        lines.append("coverage_overview  :")
        for overview in coverage_overviews:
            lines.append(
                f"  - {overview.name}: {overview.percent:.2f}% "
                f"({overview.covered} / {overview.total})"
            )
    lines.append("=" * 72)
    return "\n".join(lines)


def parse_json_list(output: str) -> list[str]:
    start = output.find("[")
    end = output.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Could not find JSON list in output")
    return json.loads(output[start : end + 1])


def parse_gtest_list(output: str) -> list[str]:
    tests: list[str] = []
    suite_name: str | None = None
    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        if (
            not line
            or line.startswith("Python Version:")
            or line.startswith("Authorized users only.")
        ):
            continue
        if not raw_line.startswith(" "):
            if line.endswith("."):
                suite_name = line[:-1]
            continue
        if suite_name is None:
            continue
        case_name = line.strip()
        if not case_name or case_name.startswith("#"):
            continue
        tests.append(f"{suite_name}.{case_name}")
    return tests


def matches_any(name: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def load_pattern_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    patterns: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def load_runtime_blacklist(path: Path = RUNTIME_BLACKLIST) -> list[str]:
    return load_pattern_file(path)


def filter_runtime_blacklist(
    tests: list[str],
    blacklist: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    patterns = load_runtime_blacklist() if blacklist is None else blacklist
    if not patterns:
        return tests, []
    kept: list[str] = []
    skipped: list[str] = []
    for test in tests:
        if matches_any(test, patterns):
            skipped.append(test)
        else:
            kept.append(test)
    return kept, skipped


def build_runtime_gtest_filter(patterns: list[str]) -> str | None:
    if not patterns:
        return None
    return ":".join(patterns)


def build_python_env(*, native_build_dir: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    pythonpath_entries = [
        *([str(native_build_dir)] if native_build_dir is not None else []),
        str(REPO_ROOT / "cinderx" / "PythonLib"),
        str(REPO_ROOT / "cinderx"),
    ]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def prepend_env(env: dict[str, str], name: str, value: str) -> None:
    existing = env.get(name)
    env[name] = value if not existing else f"{value}{os.pathsep}{existing}"


def pick_gcc_bin_dir(gcc_root: Path) -> Path:
    direct_bin = gcc_root / "bin"
    if (direct_bin / "gcc").exists():
        return direct_bin
    return gcc_root / "usr" / "bin"


def build_gcc14_env(gcc_root: Path, *, native_build_dir: Path | None = None) -> dict[str, str]:
    env = build_python_env(native_build_dir=native_build_dir)
    return add_gcc14_toolchain_env(env, gcc_root)


def add_gcc14_toolchain_env(env: dict[str, str], gcc_root: Path) -> dict[str, str]:
    bin_dir = pick_gcc_bin_dir(gcc_root)

    direct_layout = bin_dir.parent == gcc_root
    if direct_layout:
        share_man = gcc_root / "share" / "man"
        share_info = gcc_root / "share" / "info"
        lib64_dir = gcc_root / "lib64"
        lib_dir = gcc_root / "lib"
    else:
        share_man = gcc_root / "usr" / "share" / "man"
        share_info = gcc_root / "usr" / "share" / "info"
        lib64_dir = gcc_root / "usr" / "lib64"
        lib_dir = gcc_root / "usr" / "lib"
    dyninst64 = lib64_dir / "dyninst"
    dyninst32 = lib_dir / "dyninst"

    prepend_env(env, "PATH", str(bin_dir))
    prepend_env(env, "MANPATH", str(share_man))
    prepend_env(env, "INFOPATH", str(share_info))

    ld_entries = [str(lib64_dir), str(lib_dir), str(dyninst64), str(dyninst32)]
    existing_ld = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*ld_entries, *([existing_ld] if existing_ld else [])]
    )
    env["CC"] = str(bin_dir / "gcc")
    env["CXX"] = str(bin_dir / "g++")
    env["GCOV"] = str(bin_dir / "gcov")
    return env


def pythonlib_install_check_cmd(python_exe: str) -> list[str]:
    return [
        python_exe,
        "-c",
        "import cinderx, _cinderx; print(_cinderx.__file__)",
    ]


def pythonlib_pip_install_cmd(python_exe: str, *, prefix: Path | None = None) -> list[str]:
    cmd = [python_exe, "-m", "pip", "install", "--force-reinstall", "--no-deps"]
    if prefix is not None:
        cmd.extend(["--prefix", str(prefix)])
    cmd.append(".")
    return cmd


def pythonlib_install_build_base(
    repo_root: Path = REPO_ROOT,
    *,
    coverage: bool,
) -> Path:
    return repo_root / ("scratch-pythonlib-cov" if coverage else "scratch")


def pythonlib_install_prefix(output_root: Path) -> Path:
    return output_root / "pythonlib-install-prefix"


def pythonlib_install_site_packages(prefix: Path) -> Path:
    return (
        prefix
        / "lib"
        / f"python{DEFAULT_PY_VERSION}"
        / "site-packages"
    )


def pythonlib_install_build_dir(
    repo_root: Path = REPO_ROOT,
    *,
    coverage: bool,
) -> Path | None:
    candidates = sorted(
        pythonlib_install_build_base(repo_root, coverage=coverage).glob(
            f"temp.*-{sys.implementation.cache_tag}"
        )
    )
    if not candidates:
        return None
    return candidates[-1]


def pythonlib_install_env(
    gcc_root: Path,
    *,
    coverage: bool,
    prefix: Path | None = None,
) -> dict[str, str]:
    env = build_gcc14_env(gcc_root)
    if prefix is not None:
        site_packages = pythonlib_install_site_packages(prefix)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(site_packages), env["PYTHONPATH"]]
        )
    env["CINDERX_BUILD_BASE"] = str(
        pythonlib_install_build_base(REPO_ROOT, coverage=coverage)
    )
    for name, value in compute_cmake_feature_options(DEFAULT_PY_VERSION).items():
        env[name] = value
    if coverage:
        env["CFLAGS"] = "--coverage"
        env["CXXFLAGS"] = "--coverage"
        env["LDFLAGS"] = "--coverage"
    return env


def pythonlib_pip_install_env(
    gcc_root: Path,
    *,
    coverage: bool,
) -> dict[str, str]:
    env = add_gcc14_toolchain_env(dict(os.environ), gcc_root)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONNOUSERSITE", None)
    env["CINDERX_BUILD_BASE"] = str(
        pythonlib_install_build_base(REPO_ROOT, coverage=coverage)
    )
    for name, value in compute_cmake_feature_options(DEFAULT_PY_VERSION).items():
        env[name] = value
    if coverage:
        env["CFLAGS"] = "--coverage"
        env["CXXFLAGS"] = "--coverage"
        env["LDFLAGS"] = "--coverage"
    return env


def extract_pythonlib_note(output: str) -> str:
    for line in output.splitlines():
        stripped = line.strip()
        if "skipped --" in stripped:
            return stripped
        if stripped.startswith("Fatal Python error:"):
            return stripped
        if stripped.startswith("Result:"):
            return stripped
    return ""


def classify_pythonlib_result(proc: subprocess.CompletedProcess[str]) -> tuple[str, str]:
    output = proc.stdout + proc.stderr
    note = extract_pythonlib_note(output)
    if proc.returncode < 0 or proc.returncode >= 128 or "Segmentation fault" in output:
        return "CRASHED", note or "Segmentation fault"
    if "Total tests: run=0" in output:
        if "skipped" in output.lower():
            return "SKIPPED", note or "All tests skipped"
        return "NO_TESTS", note or "No tests ran"
    if "Result: SUCCESS" in output:
        return "PASSED", note
    return "FAILED", note


def extract_runtime_note(output: str) -> str:
    for line in output.splitlines():
        stripped = line.strip()
        if "Segmentation fault" in stripped:
            return stripped
        if stripped.startswith("[  FAILED  ]") or stripped.startswith("[  SKIPPED ]"):
            return stripped
    return ""


def classify_runtime_result(proc: subprocess.CompletedProcess[str]) -> tuple[str, str]:
    output = proc.stdout + proc.stderr
    note = extract_runtime_note(output)
    if proc.returncode < 0 or proc.returncode >= 128 or "Segmentation fault" in output:
        return "CRASHED", note or "Segmentation fault"
    if proc.returncode == 0:
        if "[  SKIPPED ]" in output:
            return "SKIPPED", note
        return "PASSED", note
    return "FAILED", note


def list_pythonlib_tests(args: argparse.Namespace, env: dict[str, str]) -> list[str]:
    cmd = [
        args.python_exe,
        str(TESTSCRIPTS_RUNNER),
        "dispatcher",
        "-l",
    ]
    proc = run_command(cmd, cwd=REPO_ROOT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    tests = parse_json_list(proc.stdout)
    return [test for test in tests if matches_any(test, args.filter)]


def pythonlib_module_env(module: str) -> dict[str, str]:
    if module == "test_cinderx.test_jit_frame":
        return {"PYTHONJITLIGHTWEIGHTFRAME": "0"}
    if module == "test.test_code":
        return {"CINDERX_DISABLE_SAVE_ENV_JIT_SUPPRESS": "1"}
    return {}


def clear_gcda_files(build_dir: Path) -> None:
    for gcda in build_dir.rglob("*.gcda"):
        gcda.unlink()


def run_pythonlib(
    args: argparse.Namespace, output_root: Path
) -> tuple[int, list[str], SuiteRunSummary]:
    python_dir = ensure_dir(output_root / "pythonlib")
    logs_dir = ensure_dir(python_dir / "logs")
    install_log = python_dir / "pip-install.log"
    install_build_dir: Path | None = None
    install_prefix: Path | None = None

    if args.coverage:
        install_prefix = pythonlib_install_prefix(output_root)
        if not args.no_build:
            shutil.rmtree(install_prefix, ignore_errors=True)
            install_env = pythonlib_pip_install_env(args.gcc_root, coverage=True)
            install_proc = run_command(
                pythonlib_pip_install_cmd(args.python_exe, prefix=install_prefix),
                cwd=REPO_ROOT,
                env=install_env,
            )
            write_text(install_log, install_proc.stdout + install_proc.stderr)
            if install_proc.returncode != 0:
                summary = SuiteRunSummary(
                    name="pythonlib",
                    total=0,
                    counts={"INSTALL_FAILED": 1},
                    output_dir=str(python_dir),
                    artifacts=[str(install_log)],
                )
                return install_proc.returncode, [], summary
        install_build_dir = pythonlib_install_build_dir(coverage=True)
        if not pythonlib_install_site_packages(install_prefix).exists():
            install_hint = (
                "pythonlib coverage requires an installed coverage build in the "
                "temporary prefix. Run without `--no-build`, or install with "
                "coverage flags first.\n"
            )
            write_text(install_log, install_hint)
            summary = SuiteRunSummary(
                name="pythonlib",
                total=0,
                counts={"INSTALL_REQUIRED": 1},
                output_dir=str(python_dir),
                artifacts=[str(install_log)],
            )
            return 1, [], summary
        if install_build_dir is None:
            install_hint = (
                "pythonlib coverage requires an installed coverage build. "
                "Run without `--no-build`, or install with coverage flags first.\n"
            )
            write_text(install_log, install_hint)
            summary = SuiteRunSummary(
                name="pythonlib",
                total=0,
                counts={"INSTALL_REQUIRED": 1},
                output_dir=str(python_dir),
                artifacts=[str(install_log)],
            )
            return 1, [], summary

    env = pythonlib_install_env(
        args.gcc_root,
        coverage=False,
        prefix=install_prefix,
    )

    install_check_log = python_dir / "install-check.log"
    install_check = run_command(
        pythonlib_install_check_cmd(args.python_exe),
        cwd=REPO_ROOT,
        env=env,
    )
    install_hint = (
        "pythonlib requires an installed cinderx runtime. "
        "Run `python -m pip install .` first.\n"
    )
    write_text(install_check_log, install_check.stdout + install_check.stderr + install_hint)
    if install_check.returncode != 0:
        print(f"[pythonlib] install check failed: {install_hint.strip()}")
        summary = SuiteRunSummary(
            name="pythonlib",
            total=0,
            counts={"INSTALL_REQUIRED": 1},
            output_dir=str(python_dir),
            artifacts=[str(install_check_log)],
        )
        return 1, [], summary

    tests = list_pythonlib_tests(args, env)

    write_text(python_dir / "tests.json", json.dumps(tests, indent=2))
    if args.list:
        print(json.dumps(tests, indent=2))
        summary = SuiteRunSummary(
            name="pythonlib",
            total=len(tests),
            counts={"LISTED": len(tests)},
            output_dir=str(python_dir),
            artifacts=[str(python_dir / "tests.json")],
        )
        return 0, tests, summary

    if args.coverage and install_build_dir is not None:
        clear_gcda_files(install_build_dir)

    results: list[CaseResult] = []
    for module in tests:
        safe_name = sanitize_name(module)
        log_path = logs_dir / f"{safe_name}.log"
        cmd = [args.python_exe, str(TESTSCRIPTS_RUNNER), "test", "-t", module]
        module_env = env.copy()
        module_env.update(pythonlib_module_env(module))
        proc = run_command(cmd, cwd=REPO_ROOT, env=module_env)
        write_text(log_path, proc.stdout + proc.stderr)
        status, note = classify_pythonlib_result(proc)
        results.append(
            CaseResult(
                name=module,
                status=status,
                rc=proc.returncode,
                log=str(log_path),
                note=note,
            )
        )
        print(f"[pythonlib] {module}: {status} (rc={proc.returncode})")
        if status in {"FAILED", "CRASHED"} and not args.keep_going:
            break

    write_case_results(results, python_dir, "PythonLib")
    artifacts = [
        str(python_dir / "summary.tsv"),
        str(python_dir / "summary.md"),
        str(python_dir / "tests.json"),
    ]
    if args.coverage and install_build_dir is not None:
        collect_cpp_coverage(python_dir, install_build_dir, build_gcc14_env(args.gcc_root))
        write_runtime_index(python_dir, lcov_used=False)
        artifacts.extend(
            [
                str(python_dir / "gcda-files.txt"),
                str(python_dir / "gcov-summary.txt"),
                str(python_dir / "index.md"),
            ]
        )
    if args.coverage and install_log.exists():
        artifacts.append(str(install_log))
    if install_prefix is not None:
        shutil.rmtree(install_prefix, ignore_errors=True)

    rc = 0 if all(result.status in {"PASSED", "SKIPPED", "NO_TESTS"} for result in results) else 1
    summary = SuiteRunSummary(
        name="pythonlib",
        total=len(results),
        counts=summarize_case_results(results),
        output_dir=str(python_dir),
        artifacts=artifacts,
    )
    return rc, tests, summary


def pick_runtime_build_dir(args: argparse.Namespace) -> Path:
    if args.runtime_build_dir is not None:
        return args.runtime_build_dir.resolve()
    name = "build-runtime-tests-gcc14-cov" if args.coverage else "build-runtime-tests-gcc14"
    return REPO_ROOT / name


def pick_pythonlib_build_dir(args: argparse.Namespace) -> Path:
    if args.runtime_build_dir is not None:
        return args.runtime_build_dir.resolve()
    name = "build-pythonlib-gcc14-cov" if args.coverage else "build-pythonlib-gcc14"
    return REPO_ROOT / name


def pick_product_build_dir(args: argparse.Namespace) -> Path:
    target = getattr(args, "target", "all")
    if args.coverage and target in {"all", "pythonlib"}:
        build_dir = pythonlib_install_build_dir(coverage=True)
        if build_dir is not None:
            return build_dir
    return pick_pythonlib_build_dir(args)


def pick_runtime_binary(args: argparse.Namespace, build_dir: Path) -> Path:
    if args.runtime_binary is not None:
        return args.runtime_binary.resolve()
    return build_dir / "cinderx" / "RuntimeTests" / "RuntimeTests"


def pick_runtime_cwd(args: argparse.Namespace, binary: Path) -> Path:
    if args.runtime_cwd is not None:
        return args.runtime_cwd.resolve()
    runtime_root = binary.parent / "runtime_test_root"
    return runtime_root if runtime_root.is_dir() else binary.parent


def default_gcov_inputs(repo_root: Path) -> list[Path]:
    return [
        repo_root / "cinderx" / "RuntimeTests" / "alias_class_test.cpp",
        repo_root / "cinderx" / "RuntimeTests" / "block_canonicalizer_test.cpp",
        repo_root / "cinderx" / "RuntimeTests" / "hir_test.cpp",
        repo_root / "cinderx" / "RuntimeTests" / "main.cpp",
    ]


def maybe_build_native(
    *,
    output_dir: Path,
    env: dict[str, str],
    build_dir: Path,
    build_runtime_tests: bool,
    targets: list[str],
    coverage: bool,
    skip_build: bool,
) -> int:
    if skip_build:
        return 0

    configure_cmd = [
        "cmake",
        "-S",
        str(REPO_ROOT),
        "-B",
        str(build_dir),
        f"-DPY_VERSION={DEFAULT_PY_VERSION}",
        f"-DBUILD_RUNTIME_TESTS={'ON' if build_runtime_tests else 'OFF'}",
    ]
    for name, value in compute_cmake_feature_options(DEFAULT_PY_VERSION).items():
        configure_cmd.append(f"-D{name}={value}")
    if coverage:
        configure_cmd.extend(
            [
                "-DCMAKE_C_FLAGS=--coverage",
                "-DCMAKE_CXX_FLAGS=--coverage",
                "-DCMAKE_EXE_LINKER_FLAGS=--coverage",
                "-DCMAKE_SHARED_LINKER_FLAGS=--coverage",
            ]
        )
    configure = run_command(configure_cmd, cwd=REPO_ROOT, env=env)
    write_text(output_dir / "configure.log", configure.stdout + configure.stderr)
    if configure.returncode != 0:
        return configure.returncode

    build = run_command(
        ["cmake", "--build", str(build_dir), "--target", *targets, "-j4"],
        cwd=REPO_ROOT,
        env=env,
    )
    write_text(output_dir / "build.log", build.stdout + build.stderr)
    return build.returncode


def list_runtime_tests(binary: Path, cwd: Path, env: dict[str, str], patterns: list[str]) -> list[str]:
    cmd = [str(binary), "--gtest_list_tests"]
    filter_value = build_runtime_gtest_filter(patterns)
    if filter_value is not None:
        cmd.append(f"--gtest_filter={filter_value}")
    proc = run_command(cmd, cwd=cwd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return parse_gtest_list(proc.stdout)


def write_runtime_index(runtime_dir: Path, lcov_used: bool) -> None:
    lines = [
        "# Runtime Coverage Output",
        "",
        "- `gcda-files.txt`: generated `.gcda` list",
        "- `gcov-summary.txt`: gcov stdout summary",
        "- `raw-gcov/`: generated `.gcov` files",
    ]
    if lcov_used:
        lines.extend(
            [
                "- `runtime-tests.lcov.info`: lcov coverage data",
                "- `html/`: genhtml output",
            ]
        )
    else:
        lines.append("- `html/` not generated because `lcov/genhtml` is unavailable")
    write_text(runtime_dir / "index.md", "\n".join(lines) + "\n")


def product_source_roots() -> tuple[list[Path], set[str]]:
    root = REPO_ROOT / "cinderx"
    dirs = [
        root / "Common",
        root / "CachedProperties",
        root / "Immortalize",
        root / "Interpreter",
        root / "Jit",
        root / "ParallelGC",
        root / "StaticPython",
        root / "UpstreamBorrow",
    ]
    files = {
        str(root / "_cinderx.cpp"),
        str(root / "_cinderx-lib.cpp"),
        str(root / "async_lazy_value.cpp"),
        str(root / "module_state.cpp"),
        str(root / "module_c_state.cpp"),
        str(root / "python_runtime.cpp"),
    }
    return dirs, files


def is_product_source(source: str, product_dirs: list[Path], product_files: set[str]) -> bool:
    if source in product_files:
        return True
    source_path = Path(source)
    return any(str(source_path).startswith(f"{directory}{os.sep}") for directory in product_dirs)


def parse_gcov_directory(
    gcov_dir: Path,
    product_dirs: list[Path],
    product_files: set[str],
) -> dict[str, dict[int, bool]]:
    data: dict[str, dict[int, bool]] = {}
    if not gcov_dir.exists():
        return data

    for gcov_path in gcov_dir.rglob("*.gcov"):
        source: str | None = None
        for raw in gcov_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = raw.split(":", 2)
            if len(parts) != 3:
                continue
            count_s, line_s, text = parts
            count_s = count_s.strip()
            line_s = line_s.strip()
            if line_s == "0" and text.startswith("Source:"):
                candidate = text[len("Source:") :]
                source = candidate if is_product_source(candidate, product_dirs, product_files) else None
                continue
            if source is None:
                continue
            try:
                line_no = int(line_s)
            except ValueError:
                continue
            if line_no <= 0 or count_s in {"-", "====="}:
                continue
            covered = False
            if count_s not in {"#####", "=====", "-"}:
                try:
                    covered = int(count_s.replace("*", "")) > 0
                except ValueError:
                    covered = False
            data.setdefault(source, {})
            data[source][line_no] = data[source].get(line_no, False) or covered
    return data


def build_product_baseline_gcov(
    build_dir: Path,
    env: dict[str, str],
    output_root: Path,
) -> Path:
    baseline_dir = ensure_dir(output_root / "product-baseline-gcov")
    gcno_files = [str(path) for path in build_dir.rglob("*.gcno")]
    if not gcno_files:
        write_text(output_root / "coverage-overview.log", "No .gcno files found.\n")
        return baseline_dir

    proc = run_command([env["GCOV"], *gcno_files], cwd=baseline_dir, env=env)
    write_text(output_root / "coverage-overview.log", proc.stdout + proc.stderr)
    return baseline_dir


def compute_product_coverage_overviews(
    output_root: Path,
    build_dir: Path,
    env: dict[str, str],
    included_suites: list[str],
) -> list[CoverageOverview]:
    product_dirs, product_files = product_source_roots()
    baseline_dir = build_product_baseline_gcov(build_dir, env, output_root)
    baseline = parse_gcov_directory(baseline_dir, product_dirs, product_files)
    all_lines = {(source, line_no) for source, lines in baseline.items() for line_no in lines}
    if not all_lines:
        return []

    suite_hits: dict[str, set[tuple[str, int]]] = {}
    for suite in included_suites:
        parsed = parse_gcov_directory(output_root / suite / "raw-gcov", product_dirs, product_files)
        hits: set[tuple[str, int]] = set()
        for source, lines in parsed.items():
            for line_no, covered in lines.items():
                if covered and (source, line_no) in all_lines:
                    hits.add((source, line_no))
        suite_hits[suite] = hits

    overviews: list[CoverageOverview] = []
    total = len(all_lines)
    for suite in included_suites:
        covered = len(suite_hits.get(suite, set()))
        overviews.append(
            CoverageOverview(
                name=suite,
                covered=covered,
                total=total,
                percent=(covered / total * 100.0) if total else 0.0,
            )
        )

    if len(included_suites) > 1:
        combined = set().union(*(suite_hits.get(suite, set()) for suite in included_suites))
        overviews.append(
            CoverageOverview(
                name="combined",
                covered=len(combined),
                total=total,
                percent=(len(combined) / total * 100.0) if total else 0.0,
            )
        )

    write_text(
        output_root / "coverage-overview.json",
        json.dumps([asdict(overview) for overview in overviews], indent=2),
    )
    lines = ["# Product Native Coverage Overview", ""]
    for overview in overviews:
        lines.append(
            f"- `{overview.name}`: `{overview.percent:.2f}%` "
            f"(`{overview.covered} / {overview.total}`)"
        )
    lines.append("")
    lines.append("Scope: installed `_cinderx.so` product native sources.")
    write_text(output_root / "coverage-overview.md", "\n".join(lines) + "\n")
    return overviews


def collect_cpp_coverage(
    output_dir: Path,
    build_dir: Path,
    env: dict[str, str],
) -> None:
    raw_gcov_dir = ensure_dir(output_dir / "raw-gcov")
    gcda_files = sorted(build_dir.rglob("*.gcda"))
    write_text(
        output_dir / "gcda-files.txt",
        "\n".join(str(path) for path in gcda_files) + ("\n" if gcda_files else ""),
    )

    lcov_available = (
        run_command(["bash", "-lc", "command -v lcov"], cwd=REPO_ROOT, env=env).returncode == 0
        and run_command(["bash", "-lc", "command -v genhtml"], cwd=REPO_ROOT, env=env).returncode == 0
    )

    if gcda_files:
        gcov_proc = run_command(
            [env["GCOV"], *[str(path) for path in gcda_files]],
            cwd=raw_gcov_dir,
            env=env,
        )
        write_text(output_dir / "gcov-summary.txt", gcov_proc.stdout + gcov_proc.stderr)
    else:
        write_text(output_dir / "gcov-summary.txt", "No .gcda files found.\n")

    if lcov_available:
        lcov_proc = run_command(
            [
                "lcov",
                "--capture",
                "--directory",
                str(build_dir),
                "--output-file",
                str(output_dir / "runtime-tests.lcov.info"),
            ],
            cwd=REPO_ROOT,
            env=env,
        )
        write_text(output_dir / "lcov.log", lcov_proc.stdout + lcov_proc.stderr)
        if lcov_proc.returncode == 0:
            html_dir = ensure_dir(output_dir / "html")
            genhtml_proc = run_command(
                [
                    "genhtml",
                    str(output_dir / "runtime-tests.lcov.info"),
                    "--output-directory",
                    str(html_dir),
                ],
                cwd=REPO_ROOT,
                env=env,
            )
            write_text(output_dir / "genhtml.log", genhtml_proc.stdout + genhtml_proc.stderr)

    write_runtime_index(output_dir, lcov_available)


def run_runtime(
    args: argparse.Namespace, output_root: Path
) -> tuple[int, list[str], SuiteRunSummary]:
    runtime_dir = ensure_dir(output_root / "runtime")
    logs_dir = ensure_dir(runtime_dir / "logs")
    env = build_gcc14_env(args.gcc_root)
    build_dir = pick_runtime_build_dir(args)
    binary = pick_runtime_binary(args, build_dir)

    if (not args.no_build and args.coverage) or (not args.no_build and not binary.exists()):
        build_rc = maybe_build_native(
            output_dir=runtime_dir,
            env=env,
            build_dir=build_dir,
            build_runtime_tests=True,
            targets=["RuntimeTests"],
            coverage=args.coverage,
            skip_build=args.no_build,
        )
        if build_rc != 0:
            summary = SuiteRunSummary(
                name="runtime",
                total=0,
                counts={"BUILD_FAILED": 1},
                output_dir=str(runtime_dir),
                artifacts=[
                    str(runtime_dir / "configure.log"),
                    str(runtime_dir / "build.log"),
                ],
            )
            return build_rc, [], summary

    if args.no_build and not binary.exists():
        raise FileNotFoundError(f"RuntimeTests binary not found: {binary}")

    cwd = pick_runtime_cwd(args, binary)
    tests = list_runtime_tests(binary, cwd, env, args.filter)
    tests, blacklisted = filter_runtime_blacklist(tests)
    write_text(runtime_dir / "tests.json", json.dumps(tests, indent=2))
    if blacklisted:
        write_text(runtime_dir / "blacklist-skipped.json", json.dumps(blacklisted, indent=2))
        print(
            "[runtime] blacklist skipped: "
            + ", ".join(blacklisted)
        )
    if args.list:
        print(json.dumps(tests, indent=2))
        summary = SuiteRunSummary(
            name="runtime",
            total=len(tests),
            counts={"LISTED": len(tests)},
            output_dir=str(runtime_dir),
            artifacts=[
                str(runtime_dir / "tests.json"),
                *([str(runtime_dir / "blacklist-skipped.json")] if blacklisted else []),
            ],
        )
        return 0, tests, summary

    if args.coverage:
        clear_gcda_files(build_dir)

    results: list[CaseResult] = []
    for test_name in tests:
        safe_name = sanitize_name(test_name)
        log_path = logs_dir / f"{safe_name}.log"
        proc = run_command(
            [str(binary), "--gtest_color=no", f"--gtest_filter={test_name}"],
            cwd=cwd,
            env=env,
        )
        write_text(log_path, proc.stdout + proc.stderr)
        status, note = classify_runtime_result(proc)
        results.append(
            CaseResult(
                name=test_name,
                status=status,
                rc=proc.returncode,
                log=str(log_path),
                note=note,
            )
        )
        print(f"[runtime] {test_name}: {status} (rc={proc.returncode})")
        if status in {"FAILED", "CRASHED"} and not args.keep_going:
            break

    write_case_results(results, runtime_dir, "Runtime")
    artifacts = [
        str(runtime_dir / "summary.tsv"),
        str(runtime_dir / "summary.md"),
        str(runtime_dir / "tests.json"),
        *([str(runtime_dir / "blacklist-skipped.json")] if blacklisted else []),
        str(runtime_dir / "configure.log"),
        str(runtime_dir / "build.log"),
    ]
    if args.coverage:
        collect_cpp_coverage(runtime_dir, build_dir, env)
        artifacts.extend(
            [
                str(runtime_dir / "gcda-files.txt"),
                str(runtime_dir / "gcov-summary.txt"),
                str(runtime_dir / "index.md"),
            ]
        )

    rc = 0 if all(result.status in {"PASSED", "SKIPPED"} for result in results) else 1
    summary = SuiteRunSummary(
        name="runtime",
        total=len(results),
        counts=summarize_case_results(results),
        output_dir=str(runtime_dir),
        artifacts=artifacts,
    )
    return rc, tests, summary


def ensure_product_coverage_build(
    args: argparse.Namespace,
    output_root: Path,
) -> Path:
    build_dir = pick_product_build_dir(args)
    baseline_dir = ensure_dir(output_root / "product-baseline")
    env = build_gcc14_env(args.gcc_root, native_build_dir=build_dir)

    maybe_build_native(
        output_dir=baseline_dir,
        env=env,
        build_dir=build_dir,
        build_runtime_tests=False,
        targets=["_cinderx"],
        coverage=True,
        skip_build=args.no_build,
    )
    return build_dir


def write_root_readme(output_root: Path, target: str) -> None:
    content = [
        "# UT Run Output",
        "",
        f"- target: `{target}`",
        f"- generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "- `pythonlib/`: PythonLib run outputs",
        "- `runtime/`: RuntimeTests run outputs",
        "- `-c` collects GCC C++ coverage for both test families",
    ]
    write_text(output_root / "README.md", "\n".join(content) + "\n")


def main() -> int:
    args = parse_args()
    output_root = ensure_dir(args.output.resolve())
    write_root_readme(output_root, args.target)
    print(format_start_banner(args), flush=True)

    rc = 0
    summaries: list[SuiteRunSummary] = []
    included_suites: list[str] = []
    if args.target in {"all", "pythonlib"}:
        pythonlib_rc, _, python_summary = run_pythonlib(args, output_root)
        rc = rc or pythonlib_rc
        summaries.append(python_summary)
        included_suites.append("pythonlib")

    if args.target in {"all", "runtime"}:
        runtime_rc, _, runtime_summary = run_runtime(args, output_root)
        rc = rc or runtime_rc
        summaries.append(runtime_summary)
        included_suites.append("runtime")

    coverage_overviews: list[CoverageOverview] = []
    if args.coverage and included_suites:
        product_build_dir = ensure_product_coverage_build(args, output_root)
        env = build_gcc14_env(args.gcc_root, native_build_dir=product_build_dir)
        coverage_overviews = compute_product_coverage_overviews(
            output_root=output_root,
            build_dir=product_build_dir,
            env=env,
            included_suites=included_suites,
        )

    print(
        format_finish_summary(
            args.target,
            output_root,
            summaries,
            coverage_overviews,
        ),
        flush=True,
    )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
