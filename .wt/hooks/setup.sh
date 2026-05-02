#!/usr/bin/env bash
#
# Post-create hook for worktrunk
# Creates .env.development symlink to main repo
#
# Usage: setup.sh <worktree_path> <worktree_name>
#
set -euo pipefail

WORKTREE_PATH="$1"
RAW_WORKTREE_NAME="$2"

WORKTREE_NAME=$(echo "$RAW_WORKTREE_NAME" | sed 's/^predict-rlm\.//')

# Detect layout and find main repo
BASENAME=$(basename "$WORKTREE_PATH")
if [[ "$BASENAME" == predict-rlm.* ]]; then
    REL_PATH="../predict-rlm"
else
    REL_PATH="../.."
fi

echo "==> Setting up worktree: $WORKTREE_NAME"
echo "    Worktree path: $WORKTREE_PATH"

# Symlink .env.development (shared API keys)
if [[ ! -L "$WORKTREE_PATH/.env.development" ]]; then
    rm -f "$WORKTREE_PATH/.env.development" 2>/dev/null || true
    ln -s "$REL_PATH/.env.development" "$WORKTREE_PATH/.env.development"
    echo "    Linked .env.development -> $REL_PATH/.env.development"
fi

# Symlink Claude Code settings if they exist in main repo
MAIN_REPO="$(cd "$WORKTREE_PATH/$REL_PATH" && pwd)"
if [[ -f "$MAIN_REPO/.claude/settings.local.json" ]]; then
    mkdir -p "$WORKTREE_PATH/.claude"
    if [[ ! -L "$WORKTREE_PATH/.claude/settings.local.json" ]]; then
        rm -f "$WORKTREE_PATH/.claude/settings.local.json" 2>/dev/null || true
        ln -s "../$REL_PATH/.claude/settings.local.json" "$WORKTREE_PATH/.claude/settings.local.json"
        echo "    Linked .claude/settings.local.json"
    fi
fi

echo "==> Setup complete!"
