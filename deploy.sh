#!/usr/bin/env bash
set -euo pipefail

BUMP="patch"
MESSAGE="Deploy update"

if [[ $# -gt 0 && "$1" != -* ]]; then
    BUMP="$1"
    shift
fi

while getopts "m:" opt; do
    case "$opt" in
        m)
            MESSAGE="$OPTARG"
            ;;
        *)
            echo "Usage: $0 [patch|minor|major] [-m message]" >&2
            exit 1
            ;;
    esac
done

case "$BUMP" in
    patch|minor|major)
        ;;
    *)
        echo "Invalid bump type: $BUMP" >&2
        echo "Usage: $0 [patch|minor|major] [-m message]" >&2
        exit 1
        ;;
esac

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

MESSAGE="Deploy update"

while getopts "t:m:" opt; do
  case "$opt" in
    m) MESSAGE="$OPTARG" ;;
    *) echo "Usage: $0 [-t tag] [-m message]" >&2; exit 1 ;;
  esac
done

commit_and_push_if_needed() {
  local repo_dir="$1"
  local msg="$2"

  echo "Checking $repo_dir"

  git -C "$repo_dir" status --short

  if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
    git -C "$repo_dir" add -A
    git -C "$repo_dir" commit -m "$msg"
  else
    echo "Nothing to commit in $repo_dir"
  fi

  git -C "$repo_dir" push origin HEAD
}

echo "Syncing models and metrics..."
rsync -avzh data/models/ argus:/var/www/data/models
rsync -avzh data/metrics/ argus:/var/www/data/metrics

commit_and_push_if_needed "notebooks" "$MESSAGE"
commit_and_push_if_needed "packages/forecast-core" "$MESSAGE"
commit_and_push_if_needed "." "$MESSAGE"

if [[ -n "$TAG" ]]; then
  echo "Creating tag $TAG"
  git tag -a "$TAG" -m "$MESSAGE"
  git push origin "$TAG"
fi

echo "Deploy completed."

