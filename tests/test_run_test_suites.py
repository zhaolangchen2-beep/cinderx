from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_test_suites import (  # noqa: E402
    CoverageOverview,
    SuiteRunSummary,
    build_python_env,
    build_runtime_gtest_filter,
    build_gcc14_env,
    classify_pythonlib_result,
    classify_runtime_result,
    compute_cmake_feature_options,
    default_gcov_inputs,
    default_output_root,
    filter_runtime_blacklist,
    format_finish_summary,
    format_start_banner,
    is_product_source,
    load_pattern_file,
    load_runtime_blacklist,
    pythonlib_install_build_base,
    pythonlib_install_build_dir,
    pythonlib_install_check_cmd,
    pythonlib_install_env,
    pythonlib_install_prefix,
    pythonlib_install_site_packages,
    pythonlib_pip_install_env,
    pythonlib_pip_install_cmd,
    pick_pythonlib_build_dir,
    pick_product_build_dir,
    product_source_roots,
    parse_gtest_list,
    parse_json_list,
    pick_gcc_bin_dir,
    pythonlib_module_env,
    pick_runtime_build_dir,
)


class _Proc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class ParseOutputTests(unittest.TestCase):
    def test_parse_gtest_list(self) -> None:
        output = """Python Version: 3.14
SuiteOne.
  CaseA
  CaseB
SuiteTwo.
  CaseC
"""
        self.assertEqual(
            parse_gtest_list(output),
            ["SuiteOne.CaseA", "SuiteOne.CaseB", "SuiteTwo.CaseC"],
        )

    def test_parse_json_list(self) -> None:
        output = "noise\n[\n\"test_a\",\n\"test_b\"\n]\n"
        self.assertEqual(parse_json_list(output), ["test_a", "test_b"])

    def test_build_runtime_gtest_filter(self) -> None:
        self.assertEqual(
            build_runtime_gtest_filter(["AliasClassTest.*", "BitVectorTest.*"]),
            "AliasClassTest.*:BitVectorTest.*",
        )

    def test_load_pattern_file_ignores_comments_and_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patterns.txt"
            path.write_text(
                "# comment\n\nCmdLineTest.BasicFlags\nSimplifyTest.*\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_pattern_file(path),
                ["CmdLineTest.BasicFlags", "SimplifyTest.*"],
            )

    def test_load_runtime_blacklist_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_runtime_blacklist(Path(tmp) / "missing.txt"), [])

    def test_filter_runtime_blacklist(self) -> None:
        kept, skipped = filter_runtime_blacklist(
            [
                "CmdLineTest.BasicFlags",
                "CmdLineTest.OtherFlags",
                "SimplifyTest.Foo",
                "GuardedLoadEliminationTest.Bar",
            ],
            ["CmdLineTest.BasicFlags", "SimplifyTest.*"],
        )
        self.assertEqual(
            kept,
            ["CmdLineTest.OtherFlags", "GuardedLoadEliminationTest.Bar"],
        )
        self.assertEqual(
            skipped,
            ["CmdLineTest.BasicFlags", "SimplifyTest.Foo"],
        )


class ClassificationTests(unittest.TestCase):
    def test_classify_runtime_passed(self) -> None:
        self.assertEqual(classify_runtime_result(_Proc(0, stdout="[       OK ]"))[0], "PASSED")

    def test_classify_runtime_crashed(self) -> None:
        self.assertEqual(
            classify_runtime_result(_Proc(-11, stderr="Segmentation fault"))[0],
            "CRASHED",
        )

    def test_classify_pythonlib_passed(self) -> None:
        proc = _Proc(0, stdout="Result: SUCCESS\nTotal tests: run=15 (filtered)\n")
        self.assertEqual(classify_pythonlib_result(proc)[0], "PASSED")

    def test_classify_pythonlib_skipped(self) -> None:
        proc = _Proc(
            0,
            stdout=(
                "test_cinderx.test_parallel_gc skipped -- Module not compatible with OSS imports\n"
                "Total tests: run=0 (filtered)\n"
                "Result: SUCCESS\n"
            ),
        )
        self.assertEqual(classify_pythonlib_result(proc)[0], "SKIPPED")

    def test_classify_pythonlib_crashed(self) -> None:
        proc = _Proc(139, stderr="Fatal Python error: Segmentation fault\n")
        self.assertEqual(classify_pythonlib_result(proc)[0], "CRASHED")


class PathAndEnvTests(unittest.TestCase):
    def test_default_output_root_under_cov_ut(self) -> None:
        output = default_output_root()
        self.assertEqual(output.parent.name, "ut")
        self.assertEqual(output.parent.parent.name, "cov")

    def test_build_gcc14_env_sets_expected_tools(self) -> None:
        env = build_gcc14_env(Path("/opt/openEuler/gcc-toolset-14/root"))
        self.assertTrue(env["CC"].endswith("/usr/bin/gcc"))
        self.assertTrue(env["CXX"].endswith("/usr/bin/g++"))
        self.assertTrue(env["GCOV"].endswith("/usr/bin/gcov"))
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")

    def test_build_python_env_prefers_native_build_dir(self) -> None:
        env = build_python_env(native_build_dir=Path("/tmp/native-build"))
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertTrue(env["PYTHONPATH"].split(":")[0].endswith("/tmp/native-build"))

    def test_build_gcc14_env_supports_flat_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            (bin_dir / "gcc").write_text("", encoding="utf-8")
            env = build_gcc14_env(root, native_build_dir=Path("/tmp/build"))
            self.assertEqual(env["CC"], str(bin_dir / "gcc"))
            self.assertIn("/tmp/build", env["PYTHONPATH"])

    def test_pick_gcc_bin_dir_prefers_flat_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            (bin_dir / "gcc").write_text("", encoding="utf-8")
            self.assertEqual(pick_gcc_bin_dir(root), bin_dir)

    def test_pick_pythonlib_build_dir_defaults_to_python_build(self) -> None:
        class _Args:
            runtime_build_dir = None
            coverage = False

        self.assertEqual(
            pick_pythonlib_build_dir(_Args()),
            Path(__file__).resolve().parent.parent / "build-pythonlib-gcc14",
        )

    def test_pick_runtime_build_dir_defaults_to_runtime_build(self) -> None:
        class _Args:
            runtime_build_dir = None
            coverage = False

        self.assertEqual(
            pick_runtime_build_dir(_Args()),
            Path(__file__).resolve().parent.parent / "build-runtime-tests-gcc14",
        )

    def test_pick_product_build_dir_defaults_to_python_build(self) -> None:
        class _Args:
            runtime_build_dir = None
            coverage = True

        build_dir = pick_product_build_dir(_Args())
        repo_root = Path(__file__).resolve().parent.parent
        self.assertTrue(
            build_dir == repo_root / "build-pythonlib-gcc14-cov"
            or repo_root / "scratch-pythonlib-cov" in build_dir.parents
        )

    def test_compute_cmake_feature_options_contains_expected_keys(self) -> None:
        options = compute_cmake_feature_options("3.14")
        self.assertEqual(
            set(options),
            {
                "ENABLE_ADAPTIVE_STATIC_PYTHON",
                "ENABLE_EVAL_HOOK",
                "ENABLE_INTERPRETER_LOOP",
                "ENABLE_LIGHTWEIGHT_FRAMES",
                "ENABLE_PEP523_HOOK",
            },
        )
        self.assertIn(options["ENABLE_ADAPTIVE_STATIC_PYTHON"], {"ON", "OFF"})
        self.assertIn(options["ENABLE_EVAL_HOOK"], {"ON", "OFF"})
        self.assertIn(options["ENABLE_INTERPRETER_LOOP"], {"ON", "OFF"})
        self.assertIn(options["ENABLE_LIGHTWEIGHT_FRAMES"], {"ON", "OFF"})
        self.assertIn(options["ENABLE_PEP523_HOOK"], {"ON", "OFF"})

    def test_pythonlib_install_check_cmd(self) -> None:
        cmd = pythonlib_install_check_cmd("/opt/python-3.14.3/bin/python3.14")
        self.assertEqual(cmd[:2], ["/opt/python-3.14.3/bin/python3.14", "-c"])
        self.assertIn("import cinderx, _cinderx", cmd[2])

    def test_pythonlib_pip_install_cmd(self) -> None:
        self.assertEqual(
            pythonlib_pip_install_cmd("/opt/python-3.14.3/bin/python3.14"),
            [
                "/opt/python-3.14.3/bin/python3.14",
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                ".",
            ],
        )

    def test_pythonlib_install_build_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scratch = root / "scratch-pythonlib-cov"
            scratch.mkdir()
            target = scratch / "temp.test-cpython-314"
            target.mkdir()
            self.assertEqual(pythonlib_install_build_dir(root, coverage=True), target)

    def test_pythonlib_install_build_base(self) -> None:
        root = Path("/tmp/repo")
        self.assertEqual(
            pythonlib_install_build_base(root, coverage=True),
            root / "scratch-pythonlib-cov",
        )
        self.assertEqual(
            pythonlib_install_build_base(root, coverage=False),
            root / "scratch",
        )

    def test_pythonlib_install_prefix(self) -> None:
        output_root = Path("/tmp/out")
        self.assertEqual(
            pythonlib_install_prefix(output_root),
            output_root / "pythonlib-install-prefix",
        )

    def test_pythonlib_install_site_packages(self) -> None:
        prefix = Path("/tmp/prefix")
        self.assertEqual(
            pythonlib_install_site_packages(prefix),
            prefix / "lib" / "python3.14" / "site-packages",
        )

    def test_pythonlib_install_env_coverage(self) -> None:
        env = pythonlib_install_env(Path("/opt/openEuler/gcc-toolset-14/root"), coverage=True)
        self.assertEqual(env["CFLAGS"], "--coverage")
        self.assertEqual(env["CXXFLAGS"], "--coverage")
        self.assertEqual(env["LDFLAGS"], "--coverage")
        self.assertTrue(env["CINDERX_BUILD_BASE"].endswith("scratch-pythonlib-cov"))
        self.assertIn("ENABLE_LIGHTWEIGHT_FRAMES", env)
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertIn("PYTHONPATH", env)

    def test_pythonlib_install_env_with_prefix(self) -> None:
        prefix = Path("/tmp/prefix")
        env = pythonlib_install_env(
            Path("/opt/openEuler/gcc-toolset-14/root"),
            coverage=False,
            prefix=prefix,
        )
        self.assertTrue(env["PYTHONPATH"].startswith(str(prefix / "lib" / "python3.14" / "site-packages")))

    def test_pythonlib_pip_install_env_coverage(self) -> None:
        env = pythonlib_pip_install_env(
            Path("/opt/openEuler/gcc-toolset-14/root"), coverage=True
        )
        self.assertEqual(env["CFLAGS"], "--coverage")
        self.assertEqual(env["CXXFLAGS"], "--coverage")
        self.assertEqual(env["LDFLAGS"], "--coverage")
        self.assertTrue(env["CINDERX_BUILD_BASE"].endswith("scratch-pythonlib-cov"))
        self.assertIn("ENABLE_LIGHTWEIGHT_FRAMES", env)
        self.assertNotIn("PYTHONNOUSERSITE", env)
        self.assertNotIn("PYTHONPATH", env)

    def test_pythonlib_module_env_for_jit_frame(self) -> None:
        self.assertEqual(
            pythonlib_module_env("test_cinderx.test_jit_frame"),
            {"PYTHONJITLIGHTWEIGHTFRAME": "0"},
        )

    def test_pythonlib_module_env_for_other_module(self) -> None:
        self.assertEqual(pythonlib_module_env("test.test_call"), {})

    def test_pythonlib_module_env_for_test_code(self) -> None:
        self.assertEqual(
            pythonlib_module_env("test.test_code"),
            {"CINDERX_DISABLE_SAVE_ENV_JIT_SUPPRESS": "1"},
        )

class BannerAndSummaryTests(unittest.TestCase):
    def test_format_start_banner_contains_effective_values(self) -> None:
        class _Args:
            target = "pythonlib"
            output = Path("/tmp/out")
            filter = ["test_cinderx.test_jit_exception"]
            list = False
            coverage = True
            python_exe = "/opt/python-3.14.3/bin/python3.14"
            runtime_build_dir = None
            runtime_binary = None
            runtime_cwd = None
            keep_going = True
            no_build = False
            gcc_root = Path("/opt/gcc-14")

        banner = format_start_banner(_Args())
        self.assertIn("CinderX UT task starting", banner)
        self.assertIn("target             : pythonlib", banner)
        self.assertIn("/opt/python-3.14.3/bin/python3.14", banner)
        self.assertIn("/opt/gcc-14", banner)

    def test_format_finish_summary_contains_counts_and_paths(self) -> None:
        summary = SuiteRunSummary(
            name="pythonlib",
            total=3,
            counts={"PASSED": 2, "FAILED": 1},
            output_dir="/tmp/out/pythonlib",
            artifacts=["/tmp/out/pythonlib/summary.tsv"],
        )
        text = format_finish_summary("pythonlib", Path("/tmp/out"), [summary])
        self.assertIn("CinderX UT task finished", text)
        self.assertIn("pythonlib_total      : 3", text)
        self.assertIn("PASSED=2, FAILED=1", text)
        self.assertIn("/tmp/out/pythonlib/summary.tsv", text)

    def test_format_finish_summary_contains_coverage_overview(self) -> None:
        summary = SuiteRunSummary(
            name="runtime",
            total=2,
            counts={"PASSED": 2},
            output_dir="/tmp/out/runtime",
            artifacts=[],
        )
        overview = CoverageOverview(
            name="combined",
            covered=10,
            total=20,
            percent=50.0,
        )
        text = format_finish_summary("all", Path("/tmp/out"), [summary], [overview])
        self.assertIn("coverage_overview  :", text)
        self.assertIn("combined: 50.00% (10 / 20)", text)


class ProductCoverageTests(unittest.TestCase):
    def test_is_product_source_matches_expected_paths(self) -> None:
        product_dirs, product_files = product_source_roots()
        self.assertTrue(
            is_product_source(
                str(Path(product_dirs[0]) / "foo.cpp"),
                product_dirs,
                product_files,
            )
        )
        self.assertTrue(
            is_product_source(
                next(iter(product_files)),
                product_dirs,
                product_files,
            )
        )
        self.assertFalse(
            is_product_source(
                "/tmp/not-product.cpp",
                product_dirs,
                product_files,
            )
        )

    def test_default_gcov_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self.assertEqual(
                default_gcov_inputs(repo_root),
                [
                    repo_root / "cinderx" / "RuntimeTests" / "alias_class_test.cpp",
                    repo_root / "cinderx" / "RuntimeTests" / "block_canonicalizer_test.cpp",
                    repo_root / "cinderx" / "RuntimeTests" / "hir_test.cpp",
                    repo_root / "cinderx" / "RuntimeTests" / "main.cpp",
                ],
            )


if __name__ == "__main__":
    unittest.main()
