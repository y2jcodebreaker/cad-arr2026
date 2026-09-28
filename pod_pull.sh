#!/usr/bin/env bash
# pod_pull.sh — safe `git pull` for the GPU pod. Use this instead of `git pull`.
#
# It never discards anything the pod produced. Three kinds of local change:
#   1. Tracked files edited on the pod (e.g. a sed edit to a runner)  -> STASHED, never lost
#   2. Untracked files that an incoming commit also adds              -> moved aside, pulled,
#                                                                        then compared byte-for-byte
#   3. Ignored pod output (rtsd_fullres_checkpoints/, e1_outputs/...) -> untouched: git never
#                                                                        modifies ignored files
# The pod never commits, so the pull is fast-forward only.
#
# Usage:  bash pod_pull.sh            # refuses if a run is in progress
#         bash pod_pull.sh --force    # pull anyway (the running job keeps its loaded code,
#                                     # but its ledger row will not match the new HEAD)
set -uo pipefail
cd "$(dirname "$0")"

STAMP=$(date +%Y%m%d-%H%M%S)
ASIDE=".pull_aside/$STAMP"
BRANCH=$(git branch --show-current)
MOVED=()

restore_on_failure() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ "${#MOVED[@]}" -gt 0 ]; then
    echo ""
    echo "Pull did not complete. Putting the moved-aside files back where they were:"
    for f in "${MOVED[@]}"; do
      mkdir -p "$(dirname "$f")"
      mv "$ASIDE/$f" "$f" && echo "  restored $f"
    done
  fi
  exit "$rc"
}
trap restore_on_failure EXIT

# --- 0. Never change code under a running job -------------------------------
RUNNING=$(pgrep -f "python.*(evaluate_|judge_sweep|judge_medicalization|exp3_leace|paired_quality|day1_extract|e1_|run_e1)" | tr '\n' ',' | sed 's/,$//')
if [ -n "$RUNNING" ]; then
  echo "A run is in progress:"
  ps -o pid=,args= -p "$RUNNING" | sed 's/^/  /'
  if [ "${1:-}" != "--force" ]; then
    echo "Pulling now would change the code under it. Wait for it to finish, or re-run with --force."
    exit 1
  fi
  echo "--force given: pulling anyway."
fi

git fetch --quiet origin || { echo "git fetch failed (network or credentials)."; exit 1; }

if [ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH")" ]; then
  echo "Already up to date at $(git rev-parse --short HEAD)."
  exit 0
fi

# --- 1. Tracked edits made on the pod: stash, never discard ----------------
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Tracked files were edited on this pod:"
  git status --short --untracked-files=no | sed 's/^/  /'
  git stash push --quiet -m "pod-edits-$STAMP"
  echo "  -> stashed as 'pod-edits-$STAMP'. Nothing was discarded."
  echo "     Inspect: git stash show -p stash^{/pod-edits-$STAMP}"
  echo "     Recover: git stash pop   (after the pull, if you still want the edit)"
fi

# --- 2. Untracked files an incoming commit would also add ------------------
while IFS= read -r f; do
  [ -z "$f" ] && continue
  if [ -e "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    mkdir -p "$ASIDE/$(dirname "$f")"
    mv "$f" "$ASIDE/$f"
    MOVED+=("$f")
  fi
done < <(git diff --name-only --diff-filter=A HEAD "origin/$BRANCH")

# --- 3. Fast-forward only --------------------------------------------------
if ! git -c advice.diverging=false merge --ff-only --quiet "origin/$BRANCH" 2>/dev/null; then
  echo "Fast-forward failed: this pod has commits of its own, which it should never have."
  echo "Nothing was changed. Inspect with: git log --oneline origin/$BRANCH..HEAD"
  exit 1
fi
echo "Pulled: now at $(git rev-parse --short HEAD) — $(git log -1 --format=%s)"

# --- 4. Compare what was moved aside --------------------------------------
if [ "${#MOVED[@]}" -gt 0 ]; then
  echo ""
  echo "Files the pod had that the commit also added:"
  for f in "${MOVED[@]}"; do
    if cmp -s "$ASIDE/$f" "$f"; then
      rm "$ASIDE/$f"
      echo "  identical  $f  (pod copy dropped)"
    else
      echo "  DIFFERENT  $f"
      echo "             The pod computed something the commit does not contain."
      echo "             Pod copy kept at $ASIDE/$f — a human must decide which is right."
    fi
  done
  find .pull_aside -type d -empty -delete 2>/dev/null || true
fi
MOVED=()   # pull succeeded: nothing for the trap to restore
