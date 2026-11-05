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

use crate::raw;
use crate::{MlxArray, MlxBindingsContext, MlxCError};

impl MlxBindingsContext {
    /// Applies the gap shim `mlxrs_minmax_axes` (MLX core `minmax`, the
    /// axes overload) on the runtime GPU stream, producing the minima and
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
    /// axis overload) on the runtime GPU stream, producing the minima and
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
}
