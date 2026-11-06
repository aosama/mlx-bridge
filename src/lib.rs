//! Safe ownership boundary around the official MLX C API.
//!
//! This crate owns exactly one responsibility: the MLX-C C ABI translated
//! into idiomatic Rust. It holds the bindgen-generated raw declarations, the
//! captured-error machinery, the owned handle types (`MlxArray`, `MlxDtype`,
//! `MlxArrayVector`, `MlxStream`), the per-worker bindings context
//! (`MlxBindingsContext`: submission stream and linked version), and every operation
//! wrapper family: creation, shape, padding, elementwise math, activation,
//! normalization, convolution, random, rope, quantized operations,
//! quantization construction, NVFP4, attention, the compiled-graph family,
//! the Metal kernels and capture, the general operation set, and the gap
//! shim family — this repository's own `mlxrs_*` C ABI (compiled from
//! `native_gap_shim/` by the native build engine) for MLX functions the
//! vendored MLX-C pin has not wrapped yet, so an MLX capability is
//! reachable the day the pin moves. It contains
//! no embedding policy: memory limits, the metallib path selection and
//! verification, safetensors vtable policies, and process I/O accounting
//! belong to the embedding application, which links this crate against an
//! MLX-C library and converts `MlxCError` values into its own typed runtime
//! errors at its boundary.
//!
//! # API-surface completeness
//!
//! The bridge is complete, not selective: every public function and type the
//! vendored MLX-C headers declare is carried here. The build script
//! translates the vendored tree in full and emits a bridged-symbol inventory
//! alongside the declarations, so any upstream pin bump that adds, removes,
//! or renames surface shows up as an inventory change the coverage contract
//! can fail on. The recorded state for the current pin lives in this crate's
//! `COVERAGE_INVENTORY.md`.
//!
//! Reaching into `raw` should never be necessary: `cargo xtask verify` fails
//! when a vendored function has neither a safe wrapper nor a written reason why
//! nobody needs it directly.
//!
//! Source-reference audits must never drive removals here: a wrapper with no
//! caller yet is the expected steady state of a complete bridge — the crate
//! exists precisely so that every MLX-C capability is reachable without
//! waiting for a wrapper to be written on demand.
//!
//! # Bindings-only mode
//!
//! Consumers who provide their own native MLX build — for example a pinned,
//! patched fork produced by their own native orchestration — can take this
//! crate as the bridge alone: depend with `default-features = false` and
//! the build script emits no link directives at all. The vendored-header
//! declarations, the captured-error machinery, the owned handle types, and
//! every operation wrapper family are identical; the consumer's own build
//! supplies and links the native libraries. The default `link` feature
//! carries the whole discovery ladder for everyone else.
//!
//! # Test placement
//!
//! Tests that never touch MLX-C symbols (the error-channel machinery) live
//! in this crate's test tree, except private trampolines, whose contracts sit
//! beside them because nothing outside the module can reach them. Anything exercising the bindings themselves
//! needs an MLX-C library at link time, resolved by the build script's
//! ladder: the `MLX_C_RUST_MLX_C_PREFIX` environment variable, Homebrew,
//! `pkg-config mlx-c`, and finally the pinned pair compiled into the
//! machine-shared prefix (see the build script and README). When no rung
//! resolves, library compilation and documentation builds still succeed and
//! only test binaries fail to link.

// Rustdoc only runs examples attached to items, so each guide attaches to one
// here, which is what makes every example in docs/ compile and run under
// `cargo test --doc`. Nothing here exists outside that run.
#[cfg(doctest)]
mod documentation_examples {
    #[doc = include_str!("../docs/arrays.md")]
    pub struct Arrays;

    #[doc = include_str!("../docs/closures-and-transforms.md")]
    pub struct ClosuresAndTransforms;

    #[doc = include_str!("../docs/completeness.md")]
    pub struct Completeness;

    #[doc = include_str!("../docs/distributed.md")]
    pub struct Distributed;

    #[doc = include_str!("../docs/errors.md")]
    pub struct Errors;

    #[doc = include_str!("../docs/evaluation-and-streams.md")]
    pub struct EvaluationAndStreams;

    #[doc = include_str!("../docs/exporting-functions.md")]
    pub struct ExportingFunctions;

    #[doc = include_str!("../docs/getting-started.md")]
    pub struct GettingStarted;

    #[doc = include_str!("../docs/linear-algebra.md")]
    pub struct LinearAlgebra;

    #[doc = include_str!("../docs/memory-and-runtime.md")]
    pub struct MemoryAndRuntime;

    #[doc = include_str!("../docs/metal-kernels.md")]
    pub struct MetalKernels;

    #[doc = include_str!("../docs/migration.md")]
    pub struct Migration;

    #[doc = include_str!("../docs/model-primitives.md")]
    pub struct ModelPrimitives;

    #[doc = include_str!("../docs/native-build.md")]
    pub struct NativeBuild;

    #[doc = include_str!("../docs/operations.md")]
    pub struct Operations;

    #[doc = include_str!("../docs/random.md")]
    pub struct Random;

    #[doc = include_str!("../docs/serialization.md")]
    pub struct Serialization;

    #[doc = include_str!("../docs/signal-processing.md")]
    pub struct SignalProcessing;

    #[doc = include_str!("../docs/troubleshooting.md")]
    pub struct Troubleshooting;
}

mod bindings;
mod compile;
mod metal;
mod operations;
mod runtime;
mod types;

pub mod error;
pub mod raw;

pub use bindings::context::MlxBindingsContext;
pub use compile::elementwise_graphs::MlxCompiledElementwiseGraphs;
pub use compile::gated_delta_rate_activation::MlxCompiledGatedDeltaRateActivation;
pub use compile::graph::{
    MlxGraphBuilder, array_from_vector, graph_output_array, set_graph_output,
    set_graph_output_vector,
};
pub use compile::multi_output_graph::MlxCompiledMultiOutputGraph;
pub use compile::norm_fusion::MlxCompiledDualHeadNorm;
pub use compile::swiglu::MlxCompiledSwiGlu;
pub use compile::transforms::{
    MlxCompileCache, disable_compile, enable_compile, jvp_of_closure, set_compile_mode,
    value_and_grad_of_closure, vjp_of_closure, vmap_replace, vmap_trace,
};
pub use error::{
    MlxCError, check_status, clear_captured_mlx_error, install_non_terminating_error_handler,
    take_captured_mlx_error,
};
pub use metal::cuda_kernel::{MlxCudaKernel, MlxCudaKernelConfig};
pub use metal::kernel::{
    MlxMetalKernel, MlxMetalKernelConfig, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
    apply_metal_kernel_in_graph,
};
pub use metal::metallib::set_metallib_path;
pub use operations::upstream_gap::{
    MlxEinsumPath, broadcast_shapes, einsum_path, normalize_axis_index,
};
pub use runtime::distributed_all_gather::initialize_distributed_with_all_gather;
pub use runtime::export::{
    GraphOutputFile, MlxFunctionExporter, MlxImportedFunction, MlxNodeNamer, export_function,
    export_function_with_keywords, export_graph_to_dot, print_graph,
};
pub use runtime::formatting::{MlxPrintFormat, print_format, set_print_format};
pub use runtime::io::{
    MlxIoGguf, MlxIoReader, MlxIoWriter, load_array_from_reader, load_safetensors,
    load_safetensors_from_reader, save_array_to_writer, save_safetensors,
    save_safetensors_to_writer,
};
pub use runtime::memory::{
    active_memory_bytes, array_buffer_size_bytes, cache_memory_bytes, clear_allocator_cache,
    memory_limit_bytes, peak_memory_bytes, reset_peak_memory, set_cache_limit, set_memory_limit,
    synchronize,
};
pub use runtime::platform::{
    MlxDistributedGroup, clear_distributed_backends, cuda_is_available, distributed_is_available,
    initialize_distributed, metal_is_available, metallib_path, set_wired_limit,
};
pub use runtime::stream_defaults::{
    MlxStreamThreadLocal, clear_streams, set_default_stream, synchronize_all_defaults,
};
pub use types::array::{MlxArray, MlxDtype};
pub use types::array_vector::MlxArrayVector;
pub use types::closure_custom::{MlxClosureCustom, MlxClosureCustomJvp, MlxClosureCustomVmap};
pub use types::closures::{MlxClosure, MlxClosureKwargs, MlxClosureValueAndGrad};
pub use types::device::{MlxDeviceHandle, MlxDeviceInfo};
pub use types::dtype_info::{MlxFloatInfo, result_type};
pub use types::maps::{
    MlxMapStringToArray, MlxMapStringToArrayIterator, MlxMapStringToString,
    MlxMapStringToStringIterator,
};
pub use types::stream::MlxStream;
pub use types::string::MlxString;
pub use types::vectors::{MlxVectorInt, MlxVectorStream, MlxVectorString, MlxVectorVectorArray};
