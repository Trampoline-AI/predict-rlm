#!/usr/bin/env bash
#
# Pre-remove hook for worktrunk
#
# Usage: cleanup.sh <worktree_name>
#
set -euo pipefail

WORKTREE_NAME="$1"
echo "==> Cleaning up worktree: $WORKTREE_NAME"
echo "==> Cleanup complete!"
