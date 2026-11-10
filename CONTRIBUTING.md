# Contributing to mlx-bridge

Every change to this repository flows through a pull request that links a
real issue and passes the required checks — for maintainers too. This
guide is the shortest path from idea to merged change.

## Before you open a pull request

1. **Create or pick an issue.** Every pull request must reference at least
   one real issue of this repository; the compliance check fails the pull
   request until it does.

2. **Activate the local gates** (once per clone):

   ```bash
   cargo xtask install-hooks
   ```

   This points git at `.githooks`: the pre-commit hook enforces the
   per-file size ceiling on what you stage, and the pre-push hook runs
   `cargo xtask verify` — the same lanes continuous integration runs —
   so failures surface before the push spends a GitHub Actions run.
   `MLX_BRIDGE_SKIP_PRE_PUSH=1` escapes deliberately when the failures
   are the subject under test.

3. **Verify locally** what the runners will verify:

   ```bash
   cargo xtask verify
   cargo nextest run --workspace
   cargo test --workspace --doc
   ```

   The first checks the size ceiling, coverage records, API reference,
   formatting, clippy, and documentation for the shipped
   `aarch64-apple-darwin` target; the
   second runs the test suite, one process per test, against the machine-shared
   native prefix, and the third runs the examples in `docs/`, which nextest skips
   (built automatically on first use, or through
   `cargo xtask build-native`).

## What belongs in this test suite

A contract belongs here only if a defect in this repository could make it
fail: argument order, flags and optional arguments reaching the right C
parameter, a wrapper bound to the right C function, result ordering,
ownership, the closure trampolines, the I/O vtable, error translation, and
this repository's own compiled graphs. MLX's semantics (broadcasting,
promotion, edge values, numerical accuracy, statistical quality) are proven by
MLX's own tests at the pinned release and are not restated here.

## Nobody should need a workaround

A programmer holding this crate should never have to reach into `raw` or work around a missing capability. `cargo xtask verify` and continuous integration fail when a vendored MLX-C function has neither a safe wrapper nor a written reason in `xtask/src/safe_wrapper_coverage.rs`. When the MLX pin moves, run `cargo xtask audit-upstream-surface --mlx-source <mlx checkout>` as well, which needs every public MLX function to have an MLX-C wrapper, a gap-shim entry, or a written reason in `xtask/src/audit_upstream_surface.rs`. Those registries are for functions nobody needs directly, not a backlog.

## Documentation

`docs/` holds task guides, and every Rust example in them runs under `cargo test --doc`, so an example that stops working fails the build. `docs/reference/` is generated: edit the doc comments in `src/`, then run `cargo xtask generate-api-reference`. `cargo xtask verify` fails when a public item has no doc comment or a generated page is stale. A change that adds or alters public behavior updates the relevant guide.

## Writing the pull request body

The template's `## Linked issue` section is scanned by the compliance
check ([the policy script](.github/scripts/pull-request-issue-compliance.js)):

- **Closing keywords** close the issue at merge time: `Fixes #N`,
  `Closes #N`, `Resolves #N`.
- **Non-closing references** keep the issue open: `Refs #N`.
- **Context phrases** satisfy the contract for relationships that are
  real but not closing: `part of #N`, `tracked by #N`, `umbrella #N`,
  `epic #N`, `see also #N`.
- **Groups** share one keyword: `Fixes #N, #M, and #K`.
- The same-repository issue URL is accepted wherever `#N` is.

Rejected: a bare `#N` with no keyword, references whose number is a pull
request, and numbers that do not exist as issues in this repository.

## What the required checks prove

- **Lint and document on the shipped target** — formatting, clippy with
  warnings as errors, documentation, the per-file size ceiling, and the
  pull-request policy contracts.
- **Adopt, build, and test — the one macOS journey** — the pinned MLX
  and MLX-C pair compiled from scratch and the test suite against it:
  exactly the journey an adopter's machine performs.
- **PR issue compliance** — the pull request documents its issue.

Branch protection requires all three, an up-to-date branch, and keeps
history linear: merges are squash merges.

## The safety checklist

The template's safety section is the review contract for this crate:
keep the unsafe boundary encapsulated (owned handles, checked statuses,
no raw pointers crossing the public API), reflect any MLX-C surface
change in `COVERAGE_INVENTORY.md` and the vendored header pin, and
include no credentials, personal paths, or machine-specific assumptions.

## Releases

Releases follow a written discipline — what decides a version bump,
which gates a release must pass (including the cold native journey),
the publish order, and the yank criteria. It lives in
[RELEASES.md](RELEASES.md).
