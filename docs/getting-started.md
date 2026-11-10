# Getting started

This guide takes you from an empty project to a working MLX computation in Rust.

## Requirements

You need an Apple silicon Mac, a Rust toolchain that supports edition 2024 and `rust-version` 1.101, Clang for `bindgen`, and `curl`, `tar`, and `cmake` on your `PATH`. `brew install cmake` covers the one that is usually missing.

## Add the crate

Pre-releases are never selected automatically, so name the version:

```bash
cargo add mlx-bridge@0.32.4-alpha.4
```

The first build downloads the pinned MLX and MLX-C sources, verifies their SHA-256 checksums, and compiles them once into a shared folder under `~/.cache/mlx-bridge/native/`. Every project on the machine then reuses that build. Your binaries link it statically, so they load no MLX libraries at run time, though they do read Metal kernels from that folder. See [Native build](native-build.md) for the details and alternatives.

## Your first computation

Three things happen in every program: install the error handler, create a context, and build arrays.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    // Without this, an MLX failure ends the process instead of returning an error.
    install_non_terminating_error_handler();

    // The context chooses where work runs: here, the default GPU stream.
    let context = MlxBindingsContext::new()?;

    let left = MlxArray::from_f32(&[1.0, 2.0, 3.0], &[3])?;
    let right = MlxArray::from_f32(&[10.0, 20.0, 30.0], &[3])?;

    // Nothing has been computed yet: this only records the work.
    let total = context.add(&left, &right)?;

    // Reading the values runs the graph and copies the result into a Vec.
    assert_eq!(total.to_vec_f32()?, vec![11.0, 22.0, 33.0]);
    Ok(())
}
```

## The three ideas to keep in mind

**Handles own their resources.** `MlxArray` and the other handle types release their MLX resources when dropped, and they are move-only. Use `retain` when you need a second handle to the same array.

**Operations are lazy.** Calling `context.add` builds a graph. MLX computes it when you evaluate the array or read its values. Read [Evaluation, streams, and devices](evaluation-and-streams.md) before measuring performance.

**Failures are values.** Every fallible call returns `Result<_, MlxCError>`, naming what was attempted and what MLX reported. See [Errors](errors.md).

## Where to go next

- [Arrays](arrays.md) covers every element type and how reads behave.
- [Operations](operations.md) explains the conventions all 300-plus operations share.
- [Closures and transforms](closures-and-transforms.md) covers gradients and `vmap`.
