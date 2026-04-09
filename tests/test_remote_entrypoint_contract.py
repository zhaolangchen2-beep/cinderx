from pathlib import Path


def test_remote_entrypoint_supports_skipping_arm_runtime_validation() -> None:
    text = Path("scripts/arm/remote_update_build_test.sh").read_text(encoding="utf-8")
    assert 'SKIP_ARM_RUNTIME_VALIDATION="${SKIP_ARM_RUNTIME_VALIDATION:-0}"' in text
    assert 'SKIP_ARM_RUNTIME_VALIDATION must be 0 or 1' in text
    assert 'compatibility-only validation requested; skipping pyperformance install, setup, and smoke.' in text
    assert 'VIRTUAL_ENV="$DRIVER_VENV"' in text
    assert 'PATH="$DRIVER_VENV/bin:$PATH"' in text


def test_push_to_arm_exposes_arm_runtime_validation_skip_flag() -> None:
    text = Path("scripts/push_to_arm.ps1").read_text(encoding="utf-8")
    assert '[switch]$SkipArmRuntimeValidation' in text
    assert 'SKIP_ARM_RUNTIME_VALIDATION' in text
