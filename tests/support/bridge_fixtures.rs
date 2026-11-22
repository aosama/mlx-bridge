//! Fixtures shared by the native-linked contract suites.
//!
//! Scope rule for those suites: a contract belongs here only if a defect in
//! this repository could make it fail. MLX's semantics are proven by its own
//! tests at the pinned release, so numerics appear only as evidence that
//! arguments were marshalled and results owned correctly.
//!
//! Each suite pulls this file in through `#[path]` rather than `mod.rs` so it
//! stays a plain module, not a test target of its own. Building the context
//! here makes installing the non-terminating handler impossible to forget:
//! without it an MLX failure would terminate the test process instead of
//! surfacing as the `MlxCError` these contracts assert on.

// Every suite compiles its own copy and uses a different subset.
#![allow(dead_code)]

use mlx_bridge::{
    MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxStream, cuda_is_available,
    install_non_terminating_error_handler, metal_is_available,
};

/// Single-precision comparison tolerance for values MLX computes rather than
/// stores; exact equality stays the rule for integers, booleans, and copies.
pub const FLOAT_TOLERANCE: f32 = 1e-5;

/// A context on MLX's default GPU stream, or on the CPU when the linked MLX
/// was built without any GPU backend (MLX's CPU-only build on Linux).
///
/// These suites prove the bridge's marshalling and ownership, which is the
/// same on either device, so a CPU-only build runs them rather than failing
/// them all at the first stream request. On macOS, and wherever MLX reports
/// a GPU backend, the GPU is used, so a Mac whose Metal fails still fails
/// here instead of quietly moving to the CPU.
pub fn gpu_context() -> MlxBindingsContext {
    install_non_terminating_error_handler();
    device_context().expect("the bindings context should initialize")
}

/// The context `gpu_context` returns, as a `Result`, for callbacks that run
/// inside MLX and must hand a failure back rather than panic.
pub fn device_context() -> Result<MlxBindingsContext, MlxCError> {
    if linked_mlx_has_a_gpu_backend() {
        MlxBindingsContext::new()
    } else {
        MlxBindingsContext::on_cpu()
    }
}

/// The default stream of the device `gpu_context` computes on, for graph
/// builders and stream fixtures.
pub fn device_stream() -> Result<MlxStream, MlxCError> {
    if linked_mlx_has_a_gpu_backend() {
        MlxStream::default_gpu()
    } else {
        MlxStream::default_cpu()
    }
}

/// True, after saying so on standard error, when the linked MLX has no GPU
/// backend. A contract about the GPU itself (Metal kernels, the GPU stream's
/// identity, GPU-only refusals) returns early on that build instead of
/// failing: there is no GPU for it to describe, and the same contract still
/// runs on every build that has one. Never true on macOS.
pub fn skip_without_gpu_backend(contract_name: &str) -> bool {
    if linked_mlx_has_a_gpu_backend() {
        return false;
    }
    eprintln!("skipped {contract_name}: the linked MLX was built without a GPU backend");
    true
}

/// Whether these suites treat the linked MLX as having a GPU backend.
///
/// Always true on macOS, whatever MLX reports: the crate's macOS build always
/// carries Metal, so a Mac where MLX reports no Metal is a fault the GPU
/// contracts must fail on, not a reason to skip them or move to the CPU.
/// Elsewhere it is MLX's own report of a Metal or CUDA backend.
pub fn linked_mlx_has_a_gpu_backend() -> bool {
    if cfg!(target_os = "macos") {
        return true;
    }
    let has_metal = metal_is_available().expect("MLX should report whether Metal is available");
    let has_cuda = cuda_is_available().expect("MLX should report whether CUDA is available");
    has_metal || has_cuda
}

/// A context for MLX's CPU-only families, such as linear algebra and float64.
pub fn cpu_context() -> MlxBindingsContext {
    install_non_terminating_error_handler();
    MlxBindingsContext::on_cpu().expect("the CPU bindings context should initialize")
}

pub fn float_array(values: &[f32], shape: &[i32]) -> MlxArray {
    MlxArray::from_f32(values, shape).expect("the float32 fixture should be valid")
}

pub fn int_array(values: &[i32], shape: &[i32]) -> MlxArray {
    MlxArray::from_i32(values, shape).expect("the int32 fixture should be valid")
}

pub fn floats_of(array: &MlxArray) -> Vec<f32> {
    array
        .to_vec_f32()
        .expect("the float32 array should copy back to host memory")
}

pub fn ints_of(array: &MlxArray) -> Vec<i32> {
    array
        .to_vec_i32()
        .expect("the int32 array should copy back to host memory")
}

pub fn bools_of(array: &MlxArray) -> Vec<bool> {
    array
        .to_vec_bool()
        .expect("the boolean array should copy back to host memory")
}

pub fn assert_floats_close(actual_values: &[f32], expected_values: &[f32], tolerance: f32) {
    assert_eq!(
        actual_values.len(),
        expected_values.len(),
        "element counts differ: {actual_values:?} vs {expected_values:?}"
    );
    for (actual_value, expected_value) in actual_values.iter().zip(expected_values) {
        assert!(
            (actual_value - expected_value).abs() <= tolerance,
            "expected {expected_values:?} within {tolerance}, got {actual_values:?}"
        );
    }
}

pub fn vector_of(arrays: &[&MlxArray]) -> MlxArrayVector {
    MlxArrayVector::new(arrays).expect("the array vector fixture should be valid")
}

pub fn array_at(vector: &MlxArrayVector, array_index: usize) -> MlxArray {
    vector
        .array_at(array_index, "read a fixture vector element")
        .expect("the vector element should be readable")
}

/// Asserts a failure carries both halves of the translation contract: the
/// attempted operation and MLX's own description.
pub fn assert_translated_failure(failure: &MlxCError, operation_fragment: &str) {
    assert!(
        failure.operation.contains(operation_fragment),
        "expected the operation to mention {operation_fragment:?}, got {:?}",
        failure.operation
    );
    assert!(
        !failure.description.is_empty(),
        "the failure must carry a description"
    );
}

/// A unique path under the system temporary directory for one round trip.
pub fn scratch_file_path(file_name: &str) -> String {
    let process_scoped_name = format!("mlx_bridge_{}_{file_name}", std::process::id());
    std::env::temp_dir()
        .join(process_scoped_name)
        .to_str()
        .expect("the temporary directory path should be UTF-8")
        .to_owned()
}
