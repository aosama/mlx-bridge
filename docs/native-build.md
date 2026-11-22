# Native build

The crate needs compiled MLX and MLX-C libraries. By default it builds them for you, once per machine.

## The default: automatic and static

The automatic build needs `curl`, `tar`, and `cmake` on your `PATH`. On Linux it also needs a C++ compiler and the BLAS and LAPACK development headers; see [Linux](#linux). With nothing configured and no other MLX-C found, the build script:

1. Downloads the pinned MLX and MLX-C source releases and verifies their SHA-256 checksums.
1. Compiles them with CMake, along with this repository's small gap shim, into `~/.cache/mlx-bridge/native/<identity>/prefix`.
1. Links them statically into your binary.

The identity is a hash of the pins and the build recipe, so every project on the machine using the same crate version shares one build. A from-nothing build takes a few minutes the first time and milliseconds after. Metal kernels load from the prefix through a path baked into the compiled code, so the build tree is disposable, but **binaries built this way need `mlx.metallib` to stay at that path**. The path includes your home folder, so a binary moved to another machine or user won't find it. To ship a binary elsewhere, install `mlx.metallib` where you choose and call `set_metallib_path` before the first GPU use.

Inside this repository, `cargo xtask build-native` does the same with live progress.

## The link ladder

The first match wins:

1. `MLX_BRIDGE_MLX_C_PREFIX`: a folder whose `lib/` holds the libraries.
1. Homebrew's `mlx-c`, in `/opt/homebrew` or `~/homebrew` (macOS only).
1. `pkg-config mlx-c`, requiring version 0.8.0 or newer.
1. The automatic build above.

Only `pkg-config` checks the version. If an older MLX-C sits in the first two places, linking fails with undefined symbols; see [Troubleshooting](troubleshooting.md#undefined-symbols-when-linking).

A prefix you supply may hold dynamic libraries. Their install names use `@rpath`, and a library crate cannot add an rpath to your binary, so your own build must add one, for example through a `build.rs` printing `cargo:rustc-link-arg=-Wl,-rpath,<prefix>/lib`. Otherwise the binary builds and then fails to start with "Library not loaded". Static libraries, as the automatic build produces, avoid this. The gap shim only comes with the automatic build. With your own libraries, compile `native_gap_shim/src/gap_shim.cpp` yourself to use the shim's functions, such as `stft` and `vmapped`.

## Linux

On Linux the automatic build compiles MLX's CPU backend alone: the same pins, the same static libraries and the same gap shim, without Metal. There is no GPU backend in this build, and no `mlx.metallib`.

The toolchain has two minimums. MLX-C uses C++20 `std::format`, which GCC's standard library first provides in GCC 13, so GCC 12 fails with "'format' is not a member of 'std'". bindgen parses the headers with your libclang, which must be version 16 or newer: older releases stop with "\_\_bf16 is not supported on this target". Install the `clang` package as well as `libclang-dev`: bindgen asks the `clang` executable where clang's own headers live, and without it the parse stops with "'float.h' file not found". Ubuntu 24.04 and Debian 13 meet both with their default packages. Debian 12's defaults meet neither. Its `libclang-16-dev` package fixes the parse, but its standard library is GCC 12's, so build there in a newer distribution's container.

MLX needs a BLAS and a LAPACK with their C headers (`cblas.h` and `lapacke.h`), for example the `libblas-dev`, `liblapack-dev` and `liblapacke-dev` packages on Debian and Ubuntu, or OpenBLAS. CMake finds them in the standard places. If yours are elsewhere, set CMake's own `CMAKE_PREFIX_PATH` and `CMAKE_INCLUDE_PATH` environment variables before building.

The crate links the BLAS that MLX was configured against, which it reads from the CMake package MLX installs into the prefix (`share/cmake/MLX/MLXTargets.cmake`). A static prefix you supply through `MLX_BRIDGE_MLX_C_PREFIX` must therefore be an installed MLX, with that file present.

A shared BLAS outside the loader's default directories links, but your binary will not start until the loader can find it: add an rpath to that directory in your own build (a `build.rs` printing `cargo:rustc-link-arg=-Wl,-rpath,<directory>`), or install the BLAS in a system library directory. A static BLAS links into your binary, but the libraries it depends on itself, typically `gfortran`, `gomp` or `pthread`, are not recorded by MLX's package, so link them in your own build; the build script names such an archive in a warning. Linker flags in MLX's package, such as `-pthread`, are also named in a warning rather than passed on.

Without a GPU backend there is no GPU stream, so open contexts with `MlxBindingsContext::on_cpu()` instead of `MlxBindingsContext::new()`, which the examples in these guides use and which fails on Linux. The crate's compiled graphs, such as `MlxCompiledSwiGlu`, record their operations on the GPU stream when MLX has a GPU backend and on the CPU stream when it has none. They choose this themselves, not from the context that applies them, so on a Mac they run on the GPU even when applied through a context built with `on_cpu()`.

The header parse is the same on every host and describes the Apple silicon ABI, so the float16 and bfloat16 array accessors exist only when the crate is compiled for `aarch64`.

## Bindings-only mode

If your project manages its own native image, such as a patched MLX fork, turn off the default `link` feature. The build script then emits no link directives and starts no builds, and your build links the libraries.

```toml
[dependencies]
mlx-bridge = { version = "0.32.4-alpha.6", default-features = false }
```

Declarations, wrappers, and error handling are identical.

## What is pinned

| Component | Version |
|---|---|
| MLX | `v0.32.4` |
| MLX-C | `v0.8.0` |

The crate version names the verified MLX release. MLX's tests and examples are not compiled, because consumers only link the library.

## Environment variables

| Variable | Effect |
|---|---|
| `MLX_BRIDGE_MLX_C_PREFIX` | Selects your own MLX-C prefix |
| `MLX_BRIDGE_NATIVE_PROGRESS_FILE` | Writes the automatic build's progress to a file, the only way to watch it during `cargo build` |
| `BINDGEN_EXTRA_CLANG_ARGS` | Adds arguments to the header parse, for diagnosing one-off problems |

## docs.rs and other hosts

Every host compiles and documents the crate from the vendored headers and freestanding stubs. Three kinds of build stop there, with a warning and no link flags: a host that is neither macOS nor Linux, a build for another operating system than the host (such as a Linux machine checking the `aarch64-apple-darwin` target), and a docs.rs build. Binaries link on Apple silicon macOS and on Linux.
