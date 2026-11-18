#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
    echo "Usage: $0 [patch|minor|major] [-m message] [--maintenance]"
    echo "  No arguments: run checks, generate architecture diagrams and build images."
    echo "  -m message: run checks, generate diagrams, commit and push."
    echo "  patch|minor|major with -m: also upload the release bundle and push its tag."
    echo "  --maintenance: use maintenance mode for the release."
}

BUMP=""
TAG=""
MESSAGE=""
COMMIT=false
bundle_options=()

if [[ $# -gt 0 && "$1" != -* ]]; then
    BUMP="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m)
            if [[ $# -lt 2 ]]; then
                usage >&2
                exit 1
            fi
            MESSAGE="$2"
            COMMIT=true
            shift 2
            ;;
        -m?*)
            MESSAGE="${1#-m}"
            COMMIT=true
            shift
            ;;
        --maintenance)
            bundle_options=(--maintenance)
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            usage >&2
            exit 1
            ;;
    esac
done

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
if [[ -z "$BUMP" && ${#bundle_options[@]} -gt 0 ]]; then
    echo "--maintenance requires a release bump." >&2
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
    docker build --file ".docker/$component.Dockerfile" \
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
  echo "Assembling and uploading release..."
  bundle_directory=$(mktemp -d)
  trap 'rm -rf -- "$bundle_directory"' EXIT
  ARTIFACT_ID=$(python3 scripts/deployment/release-bundle.py --output "$bundle_directory/release-bundle" "${bundle_options[@]}")
  staging="argus:/var/www/incoming/$ARTIFACT_ID/"
  rsync -avzh --mkpath "$bundle_directory/release-bundle/" "$staging"
  rsync -avzh "$bundle_directory/.release_manifest" "$bundle_directory/.release" argus:/var/www/
  printf 'Release uploaded to %s\n' "$staging"
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
