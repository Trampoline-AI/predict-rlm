# Sandbox And Research

Research feasibility before implementation whenever the RLM needs third-party
packages, external network access, heavy computation, native libraries, or
nontrivial file formats.

## Package Compatibility

The default sandbox runs Python in WASM. Packages work when they are pure Python
or available as Pyodide/Emscripten builds.

Check each candidate package:

- Does PyPI provide a `py3-none-any` wheel?
- Is the package in the Pyodide built-in package list?
- Does it depend on C extensions or native binaries?
- Is a host-side tool simpler and more reliable?

Do not assume native packages, subprocesses, system binaries, or arbitrary
filesystem access are available in the sandbox.

## Network Access

If the RLM must call external APIs from the sandbox, identify exact domains and
set `allowed_domains`. Prefer host-side tools for authenticated APIs so secrets,
refresh tokens, and provider SDKs stay outside the sandbox.

## Host-Side Tool Decisions

Use a host-side tool when the operation:

- requires authentication or private environment variables;
- calls a database, SaaS API, or internal service;
- needs a native library, subprocess, system binary, browser, or GPU;
- reads/writes outside mounted input/output files;
- is deterministic and easier to implement outside the RLM loop.

Expose concise signatures and docstrings. The RLM sees the docstring to decide
when and how to call the tool.

## Feasibility Report

Before finalizing the build plan, report:

- packages and compatibility status;
- built-in and custom skills;
- host-side tools and why they are needed;
- network allowlist domains;
- estimated iteration count and sub-LM call volume;
- input-size or chunking strategy;
- any unsupported assumptions or blockers.
