# mlx-c-rust documentation

`mlx-c-rust` is a complete, low-level Rust bridge to the MLX-C API of Apple's [MLX](https://github.com/ml-explore/mlx) array framework for Apple silicon. It gives you owned handles, typed errors, and a safe path to every MLX capability, without neural-network layers, optimizers, or model loading.

Every Rust example in these guides is compiled and run by continuous integration, so an example that stops working fails the build.

## Start here

- [Getting started](getting-started.md): install the crate and run your first computation.
- [Arrays](arrays.md): create arrays of every element type and read them back safely.
- [Evaluation, streams, and devices](evaluation-and-streams.md): lazy graphs, which device runs what, and threads.
- [Errors](errors.md): what a failure looks like and how to handle it.

## Task guides

- [Operations](operations.md): the conventions every operation shares, and where each family lives.
- [Closures and transforms](closures-and-transforms.md): gradients, `vmap`, `jvp`, `vjp`, compilation, and custom rules.
- [Linear algebra](linear-algebra.md): factorizations, solving, and norms, and why most need the CPU.
- [Signal processing](signal-processing.md): FFT, STFT, and inverse transforms.
- [Random numbers](random.md): keys, reproducibility, and distributions.
- [Model primitives](model-primitives.md): attention, RoPE, convolution, quantization, and prebuilt compiled graphs.
- [Serialization](serialization.md): safetensors, GGUF, single arrays, and custom byte transports.
- [Exporting functions](exporting-functions.md): save a traced function and replay it later.
- [Custom Metal kernels](metal-kernels.md): run your own shaders on arrays.
- [Memory and runtime](memory-and-runtime.md): limits, counters, printing, and platform queries.
- [Distributed](distributed.md): process groups and the JACCL all-gather callback.

## Building and upgrading

- [Native build](native-build.md): how the crate finds or compiles MLX, and the bindings-only mode.
- [Migrating to 0.32.4-alpha.4](migration.md): every breaking change with its replacement.
- [Troubleshooting](troubleshooting.md): symptoms, causes, and fixes.
- [Completeness and the raw layer](completeness.md): what is covered, and the few things deliberately left out.

## API reference

The reference covers every public item of the safe API, generated from the doc comments in the source. The generated `raw` declarations are left out, because they mirror the MLX-C headers:

- [Operations](reference/operations.md)
- [Types](reference/types.md)
- [Compile and transforms](reference/compile-and-transforms.md)
- [Runtime](reference/runtime.md)
- [Metal](reference/metal.md)
- [Context and errors](reference/context-and-errors.md)

The same reference is on [docs.rs](https://docs.rs/mlx-c-rust).
