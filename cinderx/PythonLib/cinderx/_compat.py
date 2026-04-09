# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilyPolicy:
    minor: str
    validated_patches: tuple[str, ...]
    default_build_patch: str
    publish_wheels: bool
    arm64_enabled_features: frozenset[str]


OSS_SUPPORTED_MINOR_FAMILIES = ("3.14", "3.15")

_FAMILY_POLICIES: dict[str, FamilyPolicy] = {
    "3.14": FamilyPolicy(
        minor="3.14",
        validated_patches=("3.14.0", "3.14.1", "3.14.2", "3.14.3"),
        default_build_patch="3.14.3",
        publish_wheels=True,
        arm64_enabled_features=frozenset(
            {"lightweight_frames"}
        ),
    ),
    "3.15": FamilyPolicy(
        minor="3.15",
        validated_patches=("3.15.0a6+",),
        default_build_patch="3.15.0a6+",
        publish_wheels=True,
        arm64_enabled_features=frozenset(),
    ),
}


def get_family_policy(minor: str) -> FamilyPolicy | None:
    return _FAMILY_POLICIES.get(minor)


def is_oss_feature_enabled(minor: str, machine: str, feature: str) -> bool:
    family = get_family_policy(minor)
    if family is None:
        return False

    if machine.lower() not in {"aarch64", "arm64"}:
        return False

    return feature in family.arm64_enabled_features
