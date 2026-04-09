import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

import cinderx.jit

from ._pyperformance_helper import find_pyperformance_benchmark


@unittest.skipUnless(cinderx.jit.is_enabled(), "Tests functionality on cinderjit")
class MdpGetCritDistExperimentTests(unittest.TestCase):
    def test_get_crit_dist_fraction_min_avoids_float_guard_deopts(self) -> None:
        module_path = find_pyperformance_benchmark("bm_mdp")
        if module_path is None:
            self.skipTest("bm_mdp benchmark source unavailable")

        code = textwrap.dedent(
            f"""
            import importlib.util
            import json
            from fractions import Fraction

            import cinderx.jit as jit
            import cinderjit

            spec = importlib.util.spec_from_file_location(
                "bm_mdp",
                {str(module_path)!r},
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            jit.enable()
            jit.enable_specialized_opcodes()
            jit.compile_after_n_calls(1000000)

            args = (50, Fraction(120, 64), 100, 100, 100, 100, 65, True, 2)
            for _ in range(10000):
                mod.getCritDist(*args)

            assert jit.force_compile(mod.getCritDist)
            counts = cinderjit.get_function_hir_opcode_counts(mod.getCritDist)
            jit.get_and_clear_runtime_stats()

            total = 0
            for _ in range(5000):
                dist = mod.getCritDist(*args)
                total += sum(v.numerator for v in dist.values())

            stats = jit.get_and_clear_runtime_stats()
            deopt_count = sum(
                entry["int"]["count"]
                for entry in stats["deopt"]
                if entry["normal"]["func_qualname"] == "getCritDist"
            )
            summary = sorted((k, v.numerator, v.denominator) for k, v in dist.items())
            print(counts.get("GuardType", 0))
            print(counts.get("CompareBool", 0))
            print(deopt_count)
            print(json.dumps(summary))
            print(total)
            """
        )

        with tempfile.TemporaryDirectory() as tmp:
            script = f"{tmp}/mdp_get_crit_dist.py"
            with open(script, "w", encoding="utf-8") as fp:
                fp.write(code)

            env_baseline = dict(os.environ)
            env_baseline["PYTHONJIT_ARM_MDP_FRACTION_MIN_COMPARE"] = "0"
            proc_baseline = subprocess.run(
                [sys.executable, script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env_baseline,
            )
            self.assertEqual(
                proc_baseline.returncode,
                0,
                f"stdout:\n{proc_baseline.stdout}\nstderr:\n{proc_baseline.stderr}",
            )

            proc_default = subprocess.run(
                [sys.executable, script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(os.environ),
            )
            self.assertEqual(
                proc_default.returncode,
                0,
                f"stdout:\n{proc_default.stdout}\nstderr:\n{proc_default.stderr}",
            )

            baseline = [
                line.strip() for line in proc_baseline.stdout.splitlines() if line.strip()
            ]
            optimized = [
                line.strip() for line in proc_default.stdout.splitlines() if line.strip()
            ]
            self.assertEqual(len(baseline), 5, proc_baseline.stdout)
            self.assertEqual(len(optimized), 5, proc_default.stdout)

            b_guard_type = int(baseline[0])
            b_compare_bool = int(baseline[1])
            b_deopt = int(baseline[2])
            b_summary = baseline[3]
            b_total = int(baseline[4])

            o_guard_type = int(optimized[0])
            o_compare_bool = int(optimized[1])
            o_deopt = int(optimized[2])
            o_summary = optimized[3]
            o_total = int(optimized[4])

            self.assertGreater(b_guard_type, 0, proc_baseline.stdout)
            self.assertEqual(b_compare_bool, 0, proc_baseline.stdout)
            self.assertEqual(b_summary, o_summary, (proc_baseline.stdout, proc_default.stdout))
            self.assertEqual(b_total, o_total, (proc_baseline.stdout, proc_default.stdout))
            self.assertGreaterEqual(o_compare_bool, 0, proc_default.stdout)
            if b_deopt > 0:
                self.assertLessEqual(
                    o_deopt,
                    b_deopt,
                    (proc_baseline.stdout, proc_default.stdout),
                )
            else:
                self.assertEqual(o_deopt, 0, (proc_baseline.stdout, proc_default.stdout))
