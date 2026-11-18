#!/usr/bin/env python3
"""Apply a prepared release to the production project."""
import subprocess
import sys
from pathlib import Path

from _rollout.applications import Applications
from _rollout.compose import Compose
from _rollout.database import Database
from _rollout.deployment import Deployment
from _rollout.proxy import NginxProxy
from _rollout.release import Release
from _rollout.state import DeploymentState


def main(root, bundle):
    compose = Compose(root)
    applications = Applications(root, compose)
    deployment = Deployment(
        compose=compose,
        release=Release(root, bundle, compose.execute),
        database=Database(root, compose),
        applications=applications,
        proxy=NginxProxy(root, compose, applications),
        state=DeploymentState(root),
    )

    deployment.apply()

if __name__ == '__main__':
    try:
        if len(sys.argv) != 3:
            raise RuntimeError('Usage: rollout.py ROOT BUNDLE')
        
        main(
            Path(sys.argv[1]).resolve(),
            Path(sys.argv[2]).resolve()
        )

    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'[deploy] Failed: {error}', file=sys.stderr)
        sys.exit(1)
