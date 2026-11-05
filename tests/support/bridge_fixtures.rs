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

use mlx_c_rust::{
    MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, install_non_terminating_error_handler,
};

/// Single-precision comparison tolerance for values MLX computes rather than
/// stores; exact equality stays the rule for integers, booleans, and copies.
pub const FLOAT_TOLERANCE: f32 = 1e-5;

pub fn gpu_context() -> MlxBindingsContext {
    install_non_terminating_error_handler();
    MlxBindingsContext::new().expect("the GPU bindings context should initialize")
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
    let process_scoped_name = format!("mlx_c_rust_{}_{file_name}", std::process::id());
    std::env::temp_dir()
        .join(process_scoped_name)
        .to_str()
        .expect("the temporary directory path should be UTF-8")
        .to_owned()
}
