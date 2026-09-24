---
name: release-prep
description: >-
  Prepare and publish a PredictRLM release from main. Use when asked to prepare,
  version, tag, or push a release. Accepts an optional release version.
metadata:
  author: Trampoline AI
  version: "1.0"
---

# Prepare a release

**Input:** optional version. Accept `0.9.0`, `0.9.0-rc3`, `0.9.0rc3`, or the
corresponding `v`-prefixed tag. Alpha and beta channels are also supported,
including Python spellings `0.9.0a3` and `0.9.0b3`. Normalize to:

- `pyproject.toml` and changelog version: hyphenated (`0.9.0-rc3`).
- Git tag: `v`-prefixed and hyphenated (`v0.9.0-rc3`).
- Distribution filename / normalized PyPI version: PEP 440 (`0.9.0rc3`).

The `Release` workflow in `.github/workflows/release.yml` compares the tag
without `v` to `[project].version` **literally**. Do not put the normalized
PEP 440 spelling in `pyproject.toml` for a hyphenated tag. Hatchling normalizes
distribution filenames but can preserve the source spelling in package metadata;
compare metadata versions using `packaging.version.Version`, not raw strings.

There is one PyPI distribution, `predict-rlm`, containing `predict_rlm`,
`rlm_gepa`, and `dspy_codex_lm`. These are not separately versioned releases.
There is no `__version__` constant to update and no npm release. `uv.lock` is
ignored and untracked; refresh it locally, but do not force-add it. Skill
metadata versions and minimum-compatible versions are not release artifacts.

Do not perform any release action until all preflight checks pass and the user
explicitly approves the proposed version. Do not switch branches, stash,
discard, commit, or otherwise repair a failed preflight. Creating or editing
this skill is not authorization to run a release.

## 1. Mandatory preflight

Run from the repository root, using Python 3.11+, `uv`, Git, and authenticated
GitHub CLI (`gh`). Run shell blocks with `bash`; `set -euo pipefail` makes a
failed check stop the block.

Before reading tags, modifying files, creating a commit or tag, or pushing:

```bash
set -euo pipefail
test "$(git branch --show-current)" = main \
  || { echo "Release preparation requires the main branch." >&2; exit 1; }
test -z "$(git status --porcelain=v1 --untracked-files=all)" \
  || { echo "Release preparation requires a completely clean worktree." >&2; exit 1; }
git remote get-url origin >/dev/null \
  || { echo "Release publication requires an origin remote." >&2; exit 1; }
```

If any check fails, stop and report it. Do not inspect tags, propose a version,
edit files, commit, create a tag, or push.

After preflight passes, run `git fetch origin --tags` so version selection and
tag collision checks include remote releases. Stop on fetch failure; never
force-update conflicting tags or automatically pull/rebase the branch.

## 2. Resolve and approve a version

Set `RELEASE_VERSION` to the exact user input, or an empty string to infer a
proposal. Whitespace is invalid. The following script performs no mutation and
prints `<tag> <source-version> <distribution-version>`:

```bash
RELEASE_VERSION='' python3 - <<'PY'
import os
import re
import subprocess

pattern = re.compile(
    r"v?(?P<major>0|[1-9][0-9]*)\.(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    r"(?:-?(?P<channel>alpha|beta|rc|a|b)(?P<serial>0|[1-9][0-9]*))?"
)
tag_pattern = re.compile(
    r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?:alpha|beta|rc)(?:0|[1-9][0-9]*))?"
)
channel_order = {"alpha": 0, "beta": 1, "rc": 2, None: 3}
tags = subprocess.run(
    ["git", "tag", "--list"], check=True, text=True, capture_output=True
).stdout.splitlines()
supplied = os.environ["RELEASE_VERSION"]
if supplied:
    match = pattern.fullmatch(supplied)
    if match is None:
        raise SystemExit("Invalid release version.")
    parts = match.groupdict()
    major, minor, patch = (int(parts[name]) for name in ("major", "minor", "patch"))
    channel = {"a": "alpha", "b": "beta"}.get(parts["channel"], parts["channel"])
    serial = int(parts["serial"] or 0)
else:
    releases = []
    for tag in tags:
        if tag_pattern.fullmatch(tag) is None:
            continue
        parts = pattern.fullmatch(tag).groupdict()
        releases.append((
            int(parts["major"]), int(parts["minor"]), int(parts["patch"]),
            channel_order[parts["channel"]], int(parts["serial"] or 0),
        ))
    if not releases:
        raise SystemExit("No supported release tags; provide an explicit version.")
    major, minor, patch, order, serial = max(releases)
    channel = {value: key for key, value in channel_order.items()}[order]
    if channel is None:
        patch += 1
    else:
        serial += 1

if channel is not None and patch != 0:
    raise SystemExit("Prereleases require patch version zero; provide a stable or minor release version.")
base = f"{major}.{minor}.{patch}"
source = base if channel is None else f"{base}-{channel}{serial}"
package_channel = {"alpha": "a", "beta": "b", "rc": "rc"}
distribution = base if channel is None else f"{base}{package_channel[channel]}{serial}"
tag = f"v{source}"
if tag in tags:
    raise SystemExit(f"Tag {tag} already exists.")
print(tag, source, distribution)
PY
```

Without input, the highest supported release tag determines the proposal:

- Stable → increment patch (`v1.2.3` → `v1.2.4`).
- Prerelease → increment serial, preserving base/channel (`v1.3.0-rc4` → `v1.3.0-rc5`).
- Numeric version ordering applies; stable outranks rc, beta, and alpha at the
  same base. Ignore unrelated tags. No supported tags → ask for a version.

Read `[project].version` and `CHANGELOG.md` before approval. Surface any mismatch
between the proposed version, current source version, and release notes. Do not
silently downgrade a source version or treat a patch proposal as appropriate
for breaking changes. Ask the user to resolve such a mismatch. If `Unreleased`
has no release-note content, stop; do not manufacture notes.

Present:

```text
Proposed release: <tag>
pyproject.toml version: <source-version>
CHANGELOG.md heading: ## [<source-version>] - <YYYY-MM-DD>
Built distribution / PyPI version: <distribution-version>
uv.lock: local refresh only; not committed
A release commit containing only pyproject.toml and CHANGELOG.md will be pushed
to origin/main, then an annotated tag will be pushed to origin. The tag triggers
a GitHub Release, package build, and PyPI publication.
Proceed?
```

Wait for an explicit affirmative response accepting the proposal. Supplying a
version alone is not approval.

## 3. Apply and verify the approved release

Re-run mandatory preflight and `git fetch origin --tags` after approval. Run the
version script with the approved version to recheck tag availability. Set `TAG`,
`SOURCE_VERSION`, and `DIST_VERSION` to its three output values; export them for
the verification scripts below. Replace `<YYYY-MM-DD>` with the release date.

Update only these two tracked release artifacts:

1. Set `[project].version` in `pyproject.toml` to `<source-version>`.
2. Move the entire non-empty release-note body under `## [Unreleased]` in
   `CHANGELOG.md` into `## [<source-version>] - <YYYY-MM-DD>`, leaving an empty
   `## [Unreleased]` above it. Preserve every note and existing historical
   release section. Keep reference-link definitions at the bottom, not in the
   release-note body.
   - Resolve the preceding published tag from Git history. Do not trust a stale
     `[Unreleased]` comparison URL or invent missing historical release notes.
   - Point `[Unreleased]` to
     `https://github.com/Trampoline-AI/predict-rlm/compare/<tag>...HEAD`.
   - Add `[<source-version>]` pointing to the comparison from the preceding
     published tag to `<tag>`. For the first release, use
     `https://github.com/Trampoline-AI/predict-rlm/releases/tag/<tag>`.

Run `uv lock` without `--upgrade` to refresh the ignored local lockfile. Do not
stage it or change dependency constraints. Verify the diff contains only the
approved version, changelog heading/links, and preserved notes. If the version
was already set to the approved value, `pyproject.toml` need not have a diff;
`CHANGELOG.md` must still change.

```bash
set -euo pipefail
git diff --check
python3 - <<'PY'
import os
import re
import subprocess
import tomllib
from pathlib import Path

source = os.environ["SOURCE_VERSION"]
assert os.environ["TAG"] == f"v{source}"
project = tomllib.loads(Path("pyproject.toml").read_text())
assert project["project"]["version"] == source, "Source version differs from approval"
lock = tomllib.loads(Path("uv.lock").read_text())
editable = [p for p in lock["package"] if p["name"] == "predict-rlm" and p["source"] == {"editable": "."}]
assert len(editable) == 1 and editable[0]["version"] == os.environ["DIST_VERSION"], "Local lock version mismatch"
notes = Path("CHANGELOG.md").read_text()
assert re.search(rf"^## \[{re.escape(source)}\] - [0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}$", notes, re.M), "Missing release heading"
assert re.search(r"^## \[Unreleased\]\s+## \[", notes, re.M), "Unreleased must be empty"
url = "https://github.com/Trampoline-AI/predict-rlm"
assert f"[Unreleased]: {url}/compare/v{source}...HEAD" in notes.splitlines()
assert any(line.startswith(f"[{source}]: {url}/") for line in notes.splitlines()), "Missing release link"
changed = set(subprocess.run(
    ["git", "diff", "HEAD", "--name-only"], check=True, text=True, capture_output=True
).stdout.splitlines())
assert "CHANGELOG.md" in changed and changed <= {"CHANGELOG.md", "pyproject.toml"}, "Unexpected release diff"
untracked = subprocess.run(
    ["git", "ls-files", "--others", "--exclude-standard"], check=True, text=True, capture_output=True
).stdout
assert not untracked, "Unexpected untracked files"
PY
uv lock --check
uv build
```

Check the newly built wheel and sdist metadata both normalize to
`<distribution-version>` using `str(packaging.version.Version(metadata_version))`.
Do not mistake old files in `dist/` for the current build. The wheel must contain
all three import packages declared in `pyproject.toml`.

Check the `Tests` workflow for the code being released. Do not claim tests passed
without observing their results; resolve failures before publication. For local
checks, use `uv run ruff check src/ tests/`, `make test-unit` (all extras), and
`make test-integration-jspi` (real Deno/WASM). Real SBX coverage separately requires
`sbx login` and `make test-integration-sbx`; report unavailable/skipped coverage.
Read `docs/runbooks/long-running-local-runs.md` before long-running local checks.

A tag identifies a commit, not uncommitted files. Create and publish the release
commit before tagging:

```bash
set -euo pipefail
git add pyproject.toml CHANGELOG.md
git commit -m "chore(predict-rlm): release $TAG"
test "$(git branch --show-current)" = main
test -z "$(git status --porcelain=v1 --untracked-files=all)"
git push origin main
python3 - <<'PY'
import subprocess

head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
remote = subprocess.check_output(
    ["git", "ls-remote", "--exit-code", "--heads", "origin", "main"], text=True
).split()
assert remote == [head, "refs/heads/main"], "origin/main does not point at the release commit"
PY
```

If verification, commit, or branch push fails, stop. Do not create or push a tag.
Never force-push or automatically amend/rebase to repair a release.

## 4. Tag, push, and verify publication

Only after verifying the release commit is on `origin/main`:

```bash
set -euo pipefail
git tag -a "$TAG" -m "Release $TAG"
git push origin "refs/tags/$TAG"
python3 - <<'PY'
import os
import subprocess


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


tag = os.environ["TAG"]
head = git("rev-parse", "HEAD")
assert git("rev-list", "-n", "1", tag) == head
assert git("ls-remote", "--exit-code", "--heads", "origin", "main").split() == [head, "refs/heads/main"]
assert git("ls-remote", "--exit-code", "origin", f"refs/tags/{tag}^{{}}").split() == [head, f"refs/tags/{tag}^{{}}"]
PY
```

The annotated tag's remote peeled SHA must match the release commit. That commit
contains the artifacts verified before committing. If tag push fails, report it
and leave the local tag intact; do not delete, move, or force-push it.

Inspect the `Release` workflow run for this tag and SHA using `gh run list
--workflow release.yml`, then `gh run watch <run-id> --exit-status` and
`gh run view <run-id>`. Check `gh release view "$TAG"` as well.

The current workflow runs **validate → GitHub Release → build → PyPI publish**.
The `publish` job uses trusted publishing in the `release` environment and may
wait for environment approval. A pushed tag or visible GitHub Release is not
proof of PyPI publication. All jobs must succeed before claiming completion.
Confirm the normalized version appears at
`https://pypi.org/pypi/predict-rlm/<distribution-version>/json`.

For a failed run, report the failing job and logs. Do not move the tag, create a
replacement version, or run a manual upload. Rerun failed jobs in the same run
only after understanding the failure; this workflow does not configure
`skip-existing`, so a partial PyPI upload is not guaranteed to retry cleanly.

Report the tag, release commit SHA, pushed remote (`origin`), GitHub Release and
workflow URLs, PyPI version, and exact verification results. Distinguish a
pushed tag, an approval-pending job, and confirmed publication.
