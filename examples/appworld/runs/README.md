# AppWorld run artifacts

This directory contains local AppWorld benchmark run artifacts. The raw run
folders include task traces, proposer traces, candidate policies, evaluator
outputs, configs, and cost logs. Treat the complete contents as protected
benchmark-derived data.

## Bundle layout

The archival bundle for this directory is:

```text
examples/appworld/runs/runs.bundle
```

`runs.bundle` is an encrypted `tar.gz` payload containing the run directories in
this folder. The canonical hosted copy is published as a Hugging Face dataset
artifact:

```text
https://huggingface.co/datasets/Trampoline-AI/predict-rlm-appworld-runs
```

## Create the bundle

From the repository root:

```bash
export RUNS_BUNDLE_PASSWORD='<password>'
COPYFILE_DISABLE=1 tar \
  --exclude='./runs.bundle' \
  --exclude='./README.md' \
  --exclude='.DS_Store' \
  --exclude='._*' \
  -C examples/appworld/runs \
  -czf - . \
  | openssl enc -aes-256-cbc -salt -pbkdf2 \
      -pass env:RUNS_BUNDLE_PASSWORD \
      -out examples/appworld/runs/runs.bundle
```

## Unpack the bundle

Download the hosted bundle if it is not already present:

```bash
make -C examples/appworld download-runs
```

Then unpack from the repository root:

```bash
export RUNS_BUNDLE_PASSWORD='<password>'
make -C examples/appworld unpack-runs
```

Equivalent direct command:

```bash
export RUNS_BUNDLE_PASSWORD='<password>'
mkdir -p examples/appworld/runs
openssl enc -d -aes-256-cbc -pbkdf2 \
  -pass env:RUNS_BUNDLE_PASSWORD \
  -in examples/appworld/runs/runs.bundle \
  | tar -xzf - -C examples/appworld/runs
```

## Hygiene

Do not publish raw extracted contents in plain text. If sharing externally,
share the encrypted bundle and only sanitized aggregate metrics or manifests in
plain text.
