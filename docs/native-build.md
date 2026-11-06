# Native build

The crate needs compiled MLX and MLX-C libraries. By default it builds them for you, once per machine.

## The default: automatic and static

The automatic build needs `curl`, `tar`, and `cmake` on your `PATH`. With nothing configured and no other MLX-C found, the build script:

1. Downloads the pinned MLX and MLX-C source releases and verifies their SHA-256 checksums.
1. Compiles them with CMake, along with this repository's small gap shim, into `~/.cache/mlx-c-rust/native/<identity>/prefix`.
1. Links them statically into your binary.

The identity is a hash of the pins and the build recipe, so every project on the machine using the same crate version shares one build. A from-nothing build takes a few minutes the first time and milliseconds after. Metal kernels load from the prefix through a path baked into the compiled code, so the build tree is disposable, but **binaries built this way need `mlx.metallib` to stay at that path**. The path includes your home folder, so a binary moved to another machine or user won't find it. To ship a binary elsewhere, install `mlx.metallib` where you choose and call `set_metallib_path` before the first GPU use.

Inside this repository, `cargo xtask build-native` does the same with live progress.

## The link ladder

The first match wins:

1. `MLX_C_RUST_MLX_C_PREFIX`: a folder whose `lib/` holds the libraries.
1. Homebrew's `mlx-c`, in `/opt/homebrew` or `~/homebrew`.
1. `pkg-config mlx-c`, requiring version 0.8.0 or newer.
1. The automatic build above.

Only `pkg-config` checks the version. If an older MLX-C sits in the first two places, linking fails with undefined symbols; see [Troubleshooting](troubleshooting.md#undefined-symbols-when-linking).

A prefix you supply may hold dynamic libraries. Their install names use `@rpath`, and a library crate cannot add an rpath to your binary, so your own build must add one, for example through a `build.rs` printing `cargo:rustc-link-arg=-Wl,-rpath,<prefix>/lib`. Otherwise the binary builds and then fails to start with "Library not loaded". Static libraries, as the automatic build produces, avoid this. The gap shim only comes with the automatic build. With your own libraries, compile `native_gap_shim/src/gap_shim.cpp` yourself to use the shim's functions, such as `stft` and `vmapped`.

## Bindings-only mode

If your project manages its own native image, such as a patched MLX fork, turn off the default `link` feature. The build script then emits no link directives and starts no builds, and your build links the libraries.

```toml
[dependencies]
mlx-c-rust = { version = "0.32.4-alpha.4", default-features = false }
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
| `MLX_C_RUST_MLX_C_PREFIX` | Selects your own MLX-C prefix |
| `MLX_C_RUST_NATIVE_PROGRESS_FILE` | Writes the automatic build's progress to a file, the only way to watch it during `cargo build` |
| `BINDGEN_EXTRA_CLANG_ARGS` | Adds arguments to the header parse, for diagnosing one-off problems |

## docs.rs and other hosts

Non-mac hosts compile and document the crate without linking, using the vendored headers and freestanding stubs, which is how docs.rs builds it. Binaries using it only link on Apple silicon.
