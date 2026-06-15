#!/usr/bin/env bash
set -euo pipefail

TAG=""
MESSAGE="Deploy update"

while getopts "t:m:" opt; do
  case "$opt" in
    t) TAG="$OPTARG" ;;
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
    git -C "$repo_dir" push origin HEAD
  else
    echo "Nothing to commit in $repo_dir"
  fi
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

