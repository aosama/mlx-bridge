# Troubleshooting

Symptoms, their causes, and what to do.

## The process exits with status 255 and prints "MLX error"

MLX's default error handler ended the process. Call `install_non_terminating_error_handler` first thing in `main`. In tests, install it in each test, because tests may run alone. See [Errors](errors.md).

## "This op is not yet supported on the GPU" or "float64 is not supported on the GPU"

Linear algebra and `float64` only run on the CPU. Use `MlxBindingsContext::on_cpu` for those operations. See [what the GPU cannot do](evaluation-and-streams.md#what-the-gpu-cannot-do).

## "There is no Stream(gpu, N) in current thread"

Streams are thread affine. Create a context on each thread, or see [Threads](evaluation-and-streams.md#threads).

## An array read fails with "array dtype is X, not Y"

Readers require the exact element type. Check `dtype()`, then call the matching reader, or convert with `astype`.

## My loop is fast, then the first read is very slow

Operations are lazy, so that read ran the whole graph. Call `evaluate` at sensible points to keep graphs small and timings honest. See [lazy evaluation](evaluation-and-streams.md#lazy-evaluation).

## A shader or graph builds fine, then fails at evaluate

Some failures only happen when the graph runs, such as Metal shader compilation. Handle errors from `evaluate` and `to_vec_*`.

## Undefined symbols when linking

Names matter here:

- `_mlx_*` such as `_mlx_matrix_transpose`, or `_mlx_distributed_group_get_rank`: an MLX-C older than 0.8 was found through `MLX_BRIDGE_MLX_C_PREFIX` or Homebrew. Update it, or remove it so the automatic build runs.
- `_mlxrs_*`: you supplied your own libraries, and the gap shim isn't among them. Compile `native_gap_shim/src/gap_shim.cpp` into your build.
- No MLX symbols resolve at all: no library was found. Look for a build-script warning, and check that `cmake` is on your `PATH`.

## The first build fails downloading or configuring

The automatic build fetches sources, and MLX's configure step fetches dependencies, so a network or GitHub failure stops it. Retry: finished phases are not repeated. A checksum failure means the download didn't match the pin, so don't bypass it.

## "automatic MLX build needs `cmake` on PATH"

Install CMake with `brew install cmake`, then rebuild.

## The binary fails to start with "Library not loaded"

You linked MLX-C dylibs from your own prefix, and their `@rpath` install names can't be resolved. A library crate cannot add an rpath to your binary, so add one in your own build, as [Native build](native-build.md#the-link-ladder) shows, or use the automatic static build.

## Metal kernels fail to load at startup

The binary looks for `mlx.metallib` at the prefix path baked in at build time. If you moved the binary to another machine or user, or deleted `~/.cache/mlx-bridge`, restore the file or call `set_metallib_path` before the first GPU use. Creating a default context counts as GPU use.

## A convolution output has the wrong shape

MLX orders weights `(output_channels, width, input_channels)` and inputs `(batch, width, channels)`. See [Convolution](model-primitives.md#convolution).

## `quantize_affine` says the group size must be 32, 64, or 128

Those are the sizes MLX accepts, and the weights' last dimension must fit them.

## A `stft` shape looks flipped

MLX orders the result `(bins, frames)`.

## Tests fail intermittently with `cargo test`

MLX's default stream, default device, and error handler are process-global. Run through `cargo nextest run`, which uses one process per test, or pass `-- --test-threads=1`.
