//! Gap-shim wrappers: the Rust path to MLX functions the vendored MLX-C
//! pin has not wrapped.
//!
//! Each wrapper calls this repository's own `mlxrs_*` C ABI (the
//! `native_gap_shim` component compiled by the native build engine) with
//! the same owned-handle and captured-error semantics as every MLX-C
//! family. The shim exists so an MLX capability is reachable the day the
//! pin moves; when an MLX-C release wraps a function, the shim entry
//! retires and the wrapper moves to the matching MLX-C family — the
//! audit-upstream-surface tool flags both the retireable entry and the
//! still-missing wrapper.

use crate::bindings::support::{c_string_argument, raw_optional_int};
use crate::error::check_status;
use crate::raw;
use crate::types::vectors::MlxVectorInt;
use crate::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxString};

/// The contraction order `einsum` would follow for one set of operands.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlxEinsumPath {
    /// Each contraction step's operand positions, in execution order.
    pub steps: Vec<Vec<i32>>,
    /// MLX's human-readable summary of the path.
    pub description: String,
}

impl MlxBindingsContext {
    /// Applies the gap shim `mlxrs_minmax_axes` (MLX core `minmax`, the
    /// axes overload) on this context's stream, producing the minima and
    /// the maxima.
    pub fn minmax_axes(
        &self,
        a: &MlxArray,
        axes: &[i32],
        keepdims: bool,
    ) -> Result<(MlxArray, MlxArray), MlxCError> {
        self.output_array_pair(
            "apply the gap shim minmax over axes",
            |minima, maxima, stream| {
                // SAFETY: Inputs and stream are live and both outputs are
                // uniquely writable.
                unsafe {
                    raw::mlxrs_minmax_axes(
                        minima,
                        maxima,
                        a.raw(),
                        axes.as_ptr(),
                        axes.len(),
                        keepdims,
                        stream,
                    )
                }
            },
        )
    }

    /// Applies the gap shim `mlxrs_minmax_axis` (MLX core `minmax`, the
    /// axis overload) on this context's stream, producing the minima and
    /// the maxima.
    pub fn minmax_axis(
        &self,
        a: &MlxArray,
        axis: i32,
        keepdims: bool,
    ) -> Result<(MlxArray, MlxArray), MlxCError> {
        self.output_array_pair(
            "apply the gap shim minmax along one axis",
            |minima, maxima, stream| {
                // SAFETY: Inputs and stream are live and both outputs are
                // uniquely writable.
                unsafe { raw::mlxrs_minmax_axis(minima, maxima, a.raw(), axis, keepdims, stream) }
            },
        )
    }

    /// The short-time Fourier transform of the last axis of `x`. Absent
    /// arguments take MLX's defaults; `onesided` defaults by input type.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the graph cannot be built.
    #[allow(clippy::too_many_arguments)]
    pub fn stft(
        &self,
        x: &MlxArray,
        n_fft: i32,
        hop_length: Option<i32>,
        win_length: Option<i32>,
        window: Option<&MlxArray>,
        center: bool,
        pad_mode: &str,
        norm: raw::mlx_fft_norm,
        onesided: Option<bool>,
    ) -> Result<MlxArray, MlxCError> {
        let operation: &'static str = "apply the gap shim stft";
        let pad_mode_argument = c_string_argument(pad_mode, operation)?;
        let mut output = MlxArray::empty();
        // SAFETY: All handles and the stream are live, the pad mode outlives
        // the call, an absent window uses the empty handle convention, and the
        // output is uniquely writable.
        let status = unsafe {
            raw::mlxrs_stft(
                output.raw_mut(),
                x.raw(),
                n_fft,
                raw_optional_int(hop_length),
                raw_optional_int(win_length),
                window.map_or_else(MlxArray::empty_raw, MlxArray::raw),
                center,
                pad_mode_argument.as_ptr(),
                norm,
                onesided.is_some(),
                onesided.unwrap_or(false),
                self.stream().raw(),
            )
        };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    /// The inverse short-time Fourier transform. Absent arguments take MLX's
    /// defaults, and `length` trims or pads the output.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the graph cannot be built.
    #[allow(clippy::too_many_arguments)]
    pub fn istft(
        &self,
        stft_matrix: &MlxArray,
        n_fft: Option<i32>,
        hop_length: Option<i32>,
        win_length: Option<i32>,
        window: Option<&MlxArray>,
        center: bool,
        norm: raw::mlx_fft_norm,
        onesided: Option<bool>,
        length: Option<i32>,
    ) -> Result<MlxArray, MlxCError> {
        let operation: &'static str = "apply the gap shim istft";
        let mut output = MlxArray::empty();
        // SAFETY: All handles and the stream are live, an absent window uses
        // the empty handle convention, and the output is uniquely writable.
        let status = unsafe {
            raw::mlxrs_istft(
                output.raw_mut(),
                stft_matrix.raw(),
                raw_optional_int(n_fft),
                raw_optional_int(hop_length),
                raw_optional_int(win_length),
                window.map_or_else(MlxArray::empty_raw, MlxArray::raw),
                center,
                norm,
                onesided.is_some(),
                onesided.unwrap_or(false),
                raw_optional_int(length),
                self.stream().raw(),
            )
        };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }
}

/// The shape two shapes broadcast to.
///
/// # Errors
/// Returns the captured MLX-C description when the shapes cannot broadcast.
pub fn broadcast_shapes(first_shape: &[i32], second_shape: &[i32]) -> Result<Vec<i32>, MlxCError> {
    let mut broadcast_shape = MlxVectorInt::empty();
    // SAFETY: Both slices outlive the call and the output handle is writable.
    let status = unsafe {
        raw::mlxrs_broadcast_shapes(
            broadcast_shape.raw_mut(),
            first_shape.as_ptr(),
            first_shape.len(),
            second_shape.as_ptr(),
            second_shape.len(),
        )
    };
    check_status(status, "broadcast MLX shapes")?;
    integers_of(&broadcast_shape)
}

/// The contraction path `einsum` would take for these subscripts and operands.
///
/// # Errors
/// Returns the captured MLX-C description when the subscripts are invalid.
pub fn einsum_path(
    subscripts: &str,
    operands: &MlxArrayVector,
) -> Result<MlxEinsumPath, MlxCError> {
    let operation: &'static str = "plan an MLX einsum path";
    let subscripts_argument = c_string_argument(subscripts, operation)?;
    let mut flattened_path = MlxVectorInt::empty();
    let mut step_lengths = MlxVectorInt::empty();
    let mut description = MlxString::empty();
    // SAFETY: The subscripts and every handle outlive the call and each output
    // handle is uniquely writable.
    let status = unsafe {
        raw::mlxrs_einsum_path(
            flattened_path.raw_mut(),
            step_lengths.raw_mut(),
            description.raw_mut(),
            subscripts_argument.as_ptr(),
            operands.raw(),
        )
    };
    check_status(status, operation)?;
    let flattened_positions = integers_of(&flattened_path)?;
    let mut steps = Vec::new();
    let mut step_start = 0_usize;
    for step_length in integers_of(&step_lengths)? {
        let step_end = step_start + usize::try_from(step_length).unwrap_or(0);
        steps.push(flattened_positions[step_start..step_end].to_vec());
        step_start = step_end;
    }
    Ok(MlxEinsumPath {
        steps,
        description: description.to_string_lossy(),
    })
}

/// The axis normalized into `[0, dimension_count)`; negative axes count from
/// the last.
///
/// # Errors
/// Returns the captured MLX-C description when the axis is out of range.
pub fn normalize_axis_index(axis: i32, dimension_count: i32) -> Result<i32, MlxCError> {
    let mut normalized_axis = 0;
    // SAFETY: The output pointer is valid writable storage.
    let status =
        unsafe { raw::mlxrs_normalize_axis_index(&mut normalized_axis, axis, dimension_count) };
    check_status(status, "normalize an MLX axis")?;
    Ok(normalized_axis)
}

fn integers_of(vector: &MlxVectorInt) -> Result<Vec<i32>, MlxCError> {
    let mut integers = Vec::with_capacity(vector.len());
    for element_index in 0..vector.len() {
        integers.push(vector.value_at(element_index)?);
    }
    Ok(integers)
}
