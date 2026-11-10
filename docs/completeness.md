# Completeness and the raw layer

The goal is that nobody holding this crate needs a workaround. This page says exactly how that is enforced and what remains out of reach.

## What is enforced

`cargo xtask verify` runs before every push, and continuous integration runs the same checks, so these fail the build:

- A vendored MLX-C function has neither a safe caller nor a written reason why nobody needs it directly.
- A public item has no doc comment, the generated [API reference](reference/operations.md) is out of date, or a guide's examples aren't attached so they run.
- `COVERAGE_INVENTORY.md` no longer matches the headers.

One more check runs by hand, because it needs an MLX checkout: `cargo xtask audit-upstream-surface --mlx-source <mlx checkout>` fails when a public MLX function has no MLX-C wrapper, no gap-shim entry, and no written reason. Run it whenever the MLX pin moves.

The counts for this release are 674 upstream MLX-C functions, 675 bridged, 85 types, and 13 gap-shim functions.

## The gap shim

Some public MLX functions have no MLX-C wrapper. `native_gap_shim/` is a small C++ library exposing them with MLX-C's conventions, under an `mlxrs_` prefix so names can't collide. When MLX-C adds a wrapper, the audit flags the shim entry for retirement. It currently covers `minmax`, `vmap`, `stft`, `istft`, `result_type`, `broadcast_shapes`, `einsum_path`, `normalize_axis_index`, `peek_default_stream`, the print settings, and `finfo`.

## The raw layer

`mlx_bridge::raw` holds every generated declaration, for anything the wrappers do differently from what you need. `MlxArray::raw` and `raw_mut` give you the underlying handle, and `check_status` translates the result.

Reaching for `raw` should be rare, and it means giving up the crate's guarantees, because `raw` calls are `unsafe` and do no element-type or layout checks. If you need it for something ordinary, that is a gap worth reporting.

## What is deliberately not exposed

| Item | Why | Instead |
|---|---|---|
| `to_stream` | Converts a C++ variant type with no C equivalent | `MlxStream`, `on_stream` |
| `precompiled_cuda_kernel` | Needs CUDA hardware; this crate ships one target, Apple silicon | None needed on this target |
| `Send` for `MlxArray` | Streams are thread affine and retained handles share state, so soundness can't be shown without an upstream guarantee | Copy with `to_vec_*` and rebuild on the other thread |
| `_mlx_array_wait` | Marks an unscheduled array available with no data behind it | `MlxArray::wait` |

## Out of scope by design

There are no neural-network modules, optimizers, or model loading, and no memory or scheduling policy. Those belong in layers above this one, such as [mlx-rs](https://github.com/oxiglade/mlx-rs).

## How it is tested

The test suite proves this repository's translation layer: argument order, flags, ownership, the closure trampolines, the I/O vtable, and error translation. MLX's numerics are proven by MLX's own tests at the pinned release. Every example in these guides also runs in continuous integration.
