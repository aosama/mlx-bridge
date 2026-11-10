# Migrating to 0.32.4-alpha.4

This release fixes safety defects, moves to MLX-C v0.8.0, and widens the safe API. Upgrade if you use keyword closures, read views, or iterate maps, because those paths could abort the process or read out of bounds in `alpha.3`. The automatic native pair rebuilds once on upgrade.

## The crate is now named `mlx-bridge`

Releases after `0.32.4-alpha.4` are published as `mlx-bridge`. Releases up to and including `0.32.4-alpha.4` stay on crates.io under the former name, `mlx-c-rust`, and receive no further updates. The API is unchanged by the rename, so moving over is a matter of new names:

| Before | After |
|---|---|
| `mlx-c-rust = "…"` in `Cargo.toml` | `mlx-bridge = "…"` |
| `use mlx_c_rust::…` | `use mlx_bridge::…` |
| `MLX_C_RUST_MLX_C_PREFIX` | `MLX_BRIDGE_MLX_C_PREFIX` |
| `MLX_C_RUST_NATIVE_PROGRESS_FILE` | `MLX_BRIDGE_NATIVE_PROGRESS_FILE` |
| `~/.cache/mlx-c-rust/native/` | `~/.cache/mlx-bridge/native/` |

The automatic native pair compiles once more into the new folder, after which you can remove `~/.cache/mlx-c-rust`. To keep both names building during a gradual move, Cargo can alias the dependency: `mlx_c_rust = { package = "mlx-bridge", version = "…" }`.

## Changes that need code edits

| Before | After | Why |
|---|---|---|
| `group.rank()` returns `i32` | Returns `Result<i32, MlxCError>` | MLX-C's query now reports failure |
| `group.size()` returns `i32` | Returns `Result<i32, MlxCError>` | Same |
| `array.strides()` returns `Vec<usize>` | Returns `Vec<i64>` | Reversed views have negative strides, which read as `usize::MAX` |
| `vmap_trace` returns three values | Returns two | The third was always empty |
| `MlxMapStringToArrayIterator` | `MlxMapStringToArrayIterator<'_>` | The iterator now borrows its map |
| `MlxMapStringToStringIterator` | `MlxMapStringToStringIterator<'_>` | Same |
| `raw::mlx_distributed_group_rank(group)` | `raw::mlx_distributed_group_get_rank(&mut rank, group)` | Renamed in MLX-C v0.8.0, as is `_size` |
| `raw::mlx_gather_qmm(...)` | Takes `global_scale` before `sorted_indices` | Added in MLX-C v0.8.0 |

Most of these only need a `?` or a type annotation.

## Calls that now behave differently

**Wrong-type reads return an error.** `to_vec_*` and `item_*` used to reinterpret memory, so `item_u32` on `1.5` returned `1069547520`, and wider reads could run out of bounds. Convert first with `astype` if you relied on a conversion.

**Views read in logical order.** `to_vec_f32` on a transpose used to return storage order, and a reversed view read past the buffer. Readers now return row-major logical order, so code that compensated for the old behavior must stop.

**Direct transfers refuse views.** `copy_evaluated_u8_values` and `copy_evaluated_u32_values` return an error for a view. Use `to_vec_*` or `copy_u32_values`.

**Missing map entries are `None`.** Map `value_at` returned an error for a missing key, and iterators returned one at the end. Both now return `Ok(None)`, so a `while let Some(...)` loop finishes cleanly. GGUF `has_metadata_*` returns `Ok(false)` for an absent key.

**Keyword closures work.** Every `MlxClosureKwargs` call used to abort with a double free.

**Some failure messages changed.** Failures from `matrix_transpose` and `unique` now name `apply MLX matrix_transpose` and `apply MLX unique`. Don't match on `operation` text unless you have to.

## Native libraries

If you supply your own MLX-C through the prefix variable, Homebrew, or `pkg-config`, you now need version 0.8. The automatic build needs no action, and no longer compiles MLX's tests and examples.

## New capabilities worth adopting

- Constructors for every element type replace casts through `f32` or `i32`.
- `MlxBindingsContext::on_cpu` and `on_stream` reach linear algebra, `float64`, and your own streams.
- `MlxClosure::vmapped`, `stft`, `istft`, `result_type`, `broadcast_shapes`, and `einsum_path` replace workarounds and `raw` calls.
- `MlxStream::on_device`, `equals`, `device`, `index`, `synchronize`, and `all_streams` were unreachable on the public type, and now work.
