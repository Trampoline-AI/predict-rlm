# Run Artifacts

These archives contain the local run artifacts used to support the
SpreadsheetBench RLM-GEPA release write-up. The extracted `runs/` directory is
gitignored, so the archives are the tracked form.

## Archives

| Archive | Contents |
| --- | --- |
| `spreadbench_verified_eval_runs_20260421-20260424.tar.gz` | Nine held-out Verified eval runs for seed and optimized skill results. |
| `spreadbench_optimize_20260417_214935.tar.gz` | Optimization run using `gpt-5.4-mini` executor, `gemini-3.0-flash` sub-LM, and `claude-opus-4-6` proposer. |
| `spreadbench_optimize_20260421_135558.tar.gz` | Optimization run whose plots back `blog_assets/score_vs_rollouts.png` and `blog_assets/candidate_lineage.png`. |
| `spreadbench_optimize_20260422_224500.tar.gz` | Additional high-validation optimization run retained for release inspection. |

## Restore

From `examples/spreadbench`:

```bash
mkdir -p runs
for archive in run_artifacts/*.tar.gz; do
  tar -xzf "$archive" -C runs
done
```

After restoring, use the normal artifact readers:

```bash
uv run rlm-gepa stats runs/<optimize-run>
uv run rlm-gepa plot runs/<optimize-run>
uv run rlm-gepa eval --dataset testset --run-dir runs/<optimize-run>
```

The archives were created with `COPYFILE_DISABLE=1` on macOS to avoid AppleDouble
metadata entries.
