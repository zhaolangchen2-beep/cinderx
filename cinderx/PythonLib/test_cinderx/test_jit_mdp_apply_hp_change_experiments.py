import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

import cinderx.jit

from ._pyperformance_helper import find_pyperformance_benchmark


@unittest.skipUnless(cinderx.jit.is_enabled(), "Tests functionality on cinderjit")
class MdpApplyHpChangeExperimentTests(unittest.TestCase):
    def test_apply_hp_change_prefers_int_clamp_path(self) -> None:
        module_path = find_pyperformance_benchmark("bm_mdp")
        if module_path is None:
            self.skipTest("bm_mdp benchmark source unavailable")

        code = textwrap.dedent(
            f"""
            import importlib.util
            import json

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

            fixed = mod.fixeddata_t(
                maxhp=100,
                stats=mod.stats_t(80, 70, 60, 50),
                lvl=50,
                badges=(False, False, False, False),
                basespeed=60,
            )
            state = mod.halfstate_t(
                fixed=fixed,
                hp=70,
                status="",
                statmods=mod.NOMODS,
                stats=fixed.stats,
            )

            for _ in range(10000):
                mod.applyHPChange(state, -3)

            assert jit.force_compile(mod.applyHPChange)
            counts = cinderjit.get_function_hir_opcode_counts(mod.applyHPChange)
            jit.get_and_clear_runtime_stats()

            total = 0
            for i in range(20000):
                total += mod.applyHPChange(state, (i % 7) - 3).hp

            stats = jit.get_and_clear_runtime_stats()
            deopt_count = sum(
                entry["int"]["count"]
                for entry in stats["deopt"]
                if entry["normal"]["func_qualname"] == "applyHPChange"
            )
            print(counts.get("GuardType", 0))
            print(counts.get("CompareBool", 0))
            print(deopt_count)
            print(total)
            """
        )

        with tempfile.TemporaryDirectory() as tmp:
            script = f"{tmp}/mdp_apply_hp_change.py"
            with open(script, "w", encoding="utf-8") as fp:
                fp.write(code)

            env_baseline = dict(os.environ)
            env_baseline["PYTHONJIT_ARM_MDP_INT_CLAMP_MIN_MAX"] = "0"
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

            baseline = [int(part) for part in proc_baseline.stdout.split()]
            optimized = [int(part) for part in proc_default.stdout.split()]
            self.assertEqual(len(baseline), 4, proc_baseline.stdout)
            self.assertEqual(len(optimized), 4, proc_default.stdout)

            b_guard_type, b_compare_bool, b_deopt, b_total = baseline
            o_guard_type, o_compare_bool, o_deopt, o_total = optimized

            self.assertGreaterEqual(b_guard_type, 0, proc_baseline.stdout)
            self.assertEqual(b_compare_bool, 0, proc_baseline.stdout)
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
