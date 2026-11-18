#!/usr/bin/env python3

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

SERVICE_IMAGES = {
    "frontend": "ghcr.io/a-kryvenko/argus-frontend",
    "api": "ghcr.io/a-kryvenko/argus-forecast",
    "clio": "ghcr.io/a-kryvenko/argus-clio",
    "prophet": "ghcr.io/a-kryvenko/argus-prophet",
    "proswin": "ghcr.io/a-kryvenko/argus-proswin",
    "intelligence": "ghcr.io/a-kryvenko/argus-intelligence",
}

def _default_files(name: str, type: str) -> list:
    return [
        f"{type}/{name}/pyproject.toml",
        f"{type}/{name}/uv.lock",
        f"{type}/{name}/src",
    ]

def _default_service_files(service: str) -> list:
    return [
        f"apps/{service}/alembic.ini"
    ] + _default_files(service, "apps")

_PACKAGES_TRACKABLE_FILES = {
    "common": _default_files("common", "packages"),
    "forecast": _default_files("forecast", "packages"),
    "forecast-core": _default_files("forecast-core", "packages"),
    "intelligence-core": _default_files("intelligence-core", "packages")
        + ["packages/intelligence-core/tools"],
}

SERVICES_TRACKABLE_FILES = {
    "frontend": [
        "apps/web",
        "package.json",
        "pnpm-lock.yaml",
        "pnpm-workspace.yaml",
        "./.docker/frontend.Dockerfile"
    ],
    "api": _default_service_files("api")
        + _PACKAGES_TRACKABLE_FILES["common"]
        + ["./.docker/api.Dockerfile"]
    ,
    "clio": _default_service_files("clio")
        + _PACKAGES_TRACKABLE_FILES["common"]
        + ["./.docker/clio.Dockerfile"]
    ,
    "prophet": _default_service_files("prophet")
        + _PACKAGES_TRACKABLE_FILES["common"]
        + _PACKAGES_TRACKABLE_FILES["forecast"]
        + _PACKAGES_TRACKABLE_FILES["forecast-core"]
        + ["./.docker/prophet.Dockerfile"]
    ,
    "proswin": [
        "apps/prophet/src",
        "packages/common/src",
        "packages/forecast/src",
        "packages/forecast-core/src/forecast_core/models/proswin.py",
        "./.docker/proswin.Dockerfile"
    ],
    "intelligence": _default_service_files("intelligence")
        + _PACKAGES_TRACKABLE_FILES["common"]
        + _PACKAGES_TRACKABLE_FILES["intelligence-core"]
        + ["./.docker/intelligence.Dockerfile"]
}

_IGNORED_ITEMS = {
    "__pycache__",
    ".pytest_cache",
    ".next",
    ".turbo",
    "node_modules",
    "test-results",
    "tests",
    ".env",
    ".env.local",
    "next-env.d.ts",
    "tsconfig.tsbuildinfo",
    "playwright.config.ts",
    "README.md",
}

_PACKAGES_TESTS = {
    "common": ["packages/common/tests"],
    "forecast": ["packages/forecast/tests"],
    "forecast-core": ["packages/forecast-core/tests"],
    "intelligence-core": ["packages/intelligence-core/tests"],
}

_SERVICES_TESTS = {
    "api": ["apps/api/tests"]
            + _PACKAGES_TESTS["common"],
    "clio": ["apps/clio/tests"]
            + _PACKAGES_TESTS["common"],
    "prophet": ["apps/prophet/tests"]
            + _PACKAGES_TESTS["common"]
            + _PACKAGES_TESTS["forecast"]
            + _PACKAGES_TESTS["forecast-core"],
    "proswin": ["apps/prophet/tests"]
            + _PACKAGES_TESTS["common"]
            + _PACKAGES_TESTS["forecast"]
            + _PACKAGES_TESTS["forecast-core"],
    "intelligence": ["apps/intelligence/tests"]
            + _PACKAGES_TESTS["common"]
            + _PACKAGES_TESTS["intelligence-core"]
}

ROOT = Path(__file__).resolve().parents[2]

def _find_important_files(p: str, files: set):
    p = ROOT / p

    if p.is_file():
        files.add(p)
        return

    for _root, _dirs, _files in os.walk(p):
        _dirs[:] = [
            d for d in _dirs
            if d not in _IGNORED_ITEMS and not d.endswith(".egg-info")
        ]

        for _f in _files:
            if _f not in _IGNORED_ITEMS:
                files.add(Path(os.path.join(_root, _f)))


def _service_hash(sources: list) -> str:
    files = set()
    for p in sources:
        _find_important_files(p, files)

    digest = hashlib.sha256()

    for p in sorted(files):
        rel = p.relative_to(ROOT)

        digest.update(rel.as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(p.read_bytes()).digest())

    return digest.hexdigest()

def changed_services(
        previous_hashes: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    current_hashes = {
        service: _service_hash(sources)
        for service, sources in SERVICES_TRACKABLE_FILES.items()
    }

    changed = [
        service
        for service, current in current_hashes.items()
        if current != previous_hashes.get(service)
    ]

    return current_hashes, changed

def image_exists(reference: str) -> bool:
    result = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", reference],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode == 0:
        return True

    error = result.stderr.strip()

    if (
        error == f"ERROR: {reference}: not found"
        or re.search(
            r": (?:manifest unknown|name unknown)(?::|$)",
            error,
            re.IGNORECASE,
        )
    ):
        return False

    raise RuntimeError(f"Cannot inspect {reference}: {error}")

if __name__ == "__main__":
    hashes, _ = changed_services({})

    services_to_rebuild = []

    for service, hash in hashes.items():
        reference = f"{SERVICE_IMAGES[service]}:src-{hash}"

        if not image_exists(reference):
            services_to_rebuild.append((service, reference, f"src-{hash}"))


    build_matrix = {
        "include": [
            {
                "service": service,
                "dockerfile": f".docker/{service}.Dockerfile",
                "image": reference,
                "image_latest": f"{SERVICE_IMAGES[service]}:latest",
                "version": version
            }
            for service, reference, version in services_to_rebuild
        ]
    }

    test_paths = sorted({
        tests
        for service, _, _ in services_to_rebuild
        if service != "frontend"
        for tests in _SERVICES_TESTS[service]
    })

    test_matrix = {
        "include": [
            {
                "project": Path(tests).parent.as_posix(),
                "tests": tests
            }
            for tests in test_paths
        ]
    }
    # To simplify Ci
    if not test_matrix["include"]:
        test_matrix["include"] = [{"project": "", "tests": ""}]

    # Outputs for Github CI outputs
    if output_file := os.getenv("GITHUB_OUTPUT"):
        with open(output_file, "a", encoding="utf-8") as output:
            output.write(f"test_matrix={json.dumps(test_matrix)}\n")
            
            output.write(f"build_matrix={json.dumps(build_matrix)}\n")
            output.write(f"has_builds={str(bool(services_to_rebuild)).lower()}\n")

        
