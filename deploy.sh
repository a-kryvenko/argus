#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
    echo "Usage: $0 [patch|minor|major] [-m message]"
    echo "  No arguments: run checks, generate architecture diagrams and build images."
    echo "  -m message: run checks, generate diagrams, commit and push."
    echo "  patch|minor|major with -m: also upload artifacts and create/push a release tag."
}

BUMP=""
TAG=""
MESSAGE=""
COMMIT=false

if [[ $# -gt 0 && "$1" != -* ]]; then
    BUMP="$1"
    shift
fi

while getopts "m:" opt; do
    case "$opt" in
        m)
            MESSAGE="$OPTARG"
            COMMIT=true
            ;;
        *)
            usage >&2
            exit 1
            ;;
    esac
done

shift "$((OPTIND - 1))"
if [[ $# -gt 0 ]]; then
    echo "Unexpected argument: $1" >&2
    usage >&2
    exit 1
fi

case "$BUMP" in
    ""|patch|minor|major)
        ;;
    *)
        echo "Invalid bump type: $BUMP" >&2
        usage >&2
        exit 1
        ;;
esac

if [[ -n "$BUMP" && "$COMMIT" = false ]]; then
    echo "A release bump requires -m message." >&2
    usage >&2
    exit 1
fi

echo "Running web lint, type checks and tests"
pnpm --filter web lint
pnpm --filter web check-types
pnpm --filter web test
pnpm --filter web test:dashboard

echo "Running Python tests"
./scripts/test-python

echo "Generating architecture diagrams"
pnpm docs:architecture

echo "Checking Docker image builds"
for component in frontend api clio prophet proswin intelligence; do
    docker build --file ".deploy/$component.Dockerfile" \
        --tag "argus-$component:build-check" \
        --build-arg DEPLOYMENT_VERSION=build-check .
done

if [[ "$COMMIT" = false ]]; then
    echo "All checks passed, architecture diagrams generated and images built."
    exit 0
fi

if [[ -n "$BUMP" ]]; then
    LAST_TAG=$(git tag --sort=-v:refname | head -n 1)

    if [[ -z "$LAST_TAG" ]]; then
        MAJOR=0
        MINOR=0
        PATCH=0
    else
        VERSION="${LAST_TAG#v}"

        IFS='.' read -r MAJOR MINOR PATCH <<< "$VERSION"
    fi

    case "$BUMP" in
        patch)
            PATCH=$((PATCH + 1))
            ;;
        minor)
            MINOR=$((MINOR + 1))
            PATCH=0
            ;;
        major)
            MAJOR=$((MAJOR + 1))
            MINOR=0
            PATCH=0
            ;;
    esac

    TAG="v${MAJOR}.${MINOR}.${PATCH}"

    echo "$LAST_TAG -> $TAG"
fi

echo "$MESSAGE"

commit_and_push_if_needed() {
  local repo_dir="$1"
  local msg="$2"

  echo "Checking $repo_dir"

  git -C "$repo_dir" status --short

  if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
    git -C "$repo_dir" add -A
    if git -C "$repo_dir" diff --cached --quiet; then
      echo "Nothing staged to commit in $repo_dir"
    else
      git -C "$repo_dir" commit -m "$msg"
    fi
  else
    echo "Nothing to commit in $repo_dir"
  fi

  git -C "$repo_dir" push origin HEAD
}

if [[ -n "$BUMP" ]]; then
  echo "Syncing models, metrics and demo..."
  ARTIFACT_ID=$(python3 scripts/deployment/release.py artifacts --output .deploy/artifacts.json)
  for directory in models metrics; do
    rsync -avzh --rsync-path="mkdir -p /var/www/artifacts/$ARTIFACT_ID/$directory && rsync" \
      "data/$directory/" "argus:/var/www/artifacts/$ARTIFACT_ID/$directory/"
  done
  rsync -avzh --rsync-path="mkdir -p /var/www/artifacts/$ARTIFACT_ID/demo && rsync" \
    data/demo/current.json "argus:/var/www/artifacts/$ARTIFACT_ID/demo/current.json"
fi

commit_and_push_if_needed "notebooks" "$MESSAGE"
commit_and_push_if_needed "packages/forecast-core" "$MESSAGE"
commit_and_push_if_needed "packages/intelligence-core" "$MESSAGE"
commit_and_push_if_needed "." "$MESSAGE"

if [[ -n "$TAG" ]]; then
  echo "Creating tag $TAG"
  git tag -a "$TAG" -m "$MESSAGE"
  git push origin "$TAG"
  echo "Deploy completed."
else
  echo "Commit and push completed."
fi
