//! Validated wrappers over MLX's fused scaled dot-product attention.
//!
//! C ABI: `mlx-c/mlx/c/fast.h::mlx_fast_scaled_dot_product_attention`.
//! C++ bridge: `mlx-c/mlx/c/fast.cpp`, forwarding to
//! `mlx::core::fast::scaled_dot_product_attention`. MLX computes attention as
//! `softmax(scale * Q @ K^T + mask) @ V` and performs softmax in Float32 even for
//! BF16 inputs. Model code should shape/pad/segment tensors, not reproduce this
//! math. Callers can select unmasked, causal, or explicit array-mask execution.
//!
//! The windowed entries reach MLX's sliding window through the gap shim
//! (`mlxrs_fast_scaled_dot_product_attention_windowed`), because MLX-C v0.8.0
//! predates the `window_size` parameter MLX v0.32.5 added.

use crate::MlxBindingsContext;
use crate::MlxCError;
use crate::{MlxArray, raw};

impl MlxBindingsContext {
    /// Applies MLX-C fused unmasked attention over `[batch, heads, length, width]`.
    ///
    /// Q heads may be a multiple of K/V heads for grouped-query attention.
    pub fn scaled_dot_product_attention(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        scale: f32,
    ) -> Result<MlxArray, MlxCError> {
        validate_attention_arguments(queries, keys, values, scale)?;
        self.scaled_dot_product_attention_with_mode(
            queries,
            keys,
            values,
            scale,
            c"".as_ptr(),
            MlxArray::empty_raw(),
            "apply unmasked scaled dot-product attention",
        )
    }

    /// Applies causal scaled dot-product attention without an explicit mask array.
    pub fn causal_scaled_dot_product_attention(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        scale: f32,
    ) -> Result<MlxArray, MlxCError> {
        validate_attention_arguments(queries, keys, values, scale)?;
        self.scaled_dot_product_attention_with_mode(
            queries,
            keys,
            values,
            scale,
            c"causal".as_ptr(),
            MlxArray::empty_raw(),
            "apply causal scaled dot-product attention",
        )
    }

    /// Applies fused attention with a broadcast-compatible boolean or additive mask.
    pub fn masked_scaled_dot_product_attention(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        scale: f32,
        mask: &MlxArray,
    ) -> Result<MlxArray, MlxCError> {
        validate_masked_attention_arguments(queries, keys, values, scale, mask)?;
        self.scaled_dot_product_attention_with_mode(
            queries,
            keys,
            values,
            scale,
            c"array".as_ptr(),
            mask.raw(),
            "apply masked scaled dot-product attention",
        )
    }

    /// Applies attention in which each query sees only the keys fewer than
    /// `window_size` positions away, in either direction.
    ///
    /// Positions align at the end of the sequences: the last query sits at the
    /// last key, which is how incremental decoding lays out a key cache that
    /// is longer than the new queries.
    ///
    /// # Errors
    /// Fails when `window_size` is zero or larger than `i32::MAX`, and for the
    /// argument errors [`Self::scaled_dot_product_attention`] reports.
    pub fn windowed_scaled_dot_product_attention(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        scale: f32,
        window_size: u32,
    ) -> Result<MlxArray, MlxCError> {
        const OPERATION: &str = "apply windowed scaled dot-product attention";
        validate_attention_arguments(queries, keys, values, scale)?;
        let window = AttentionWindow::new(window_size, OPERATION)?;
        self.windowed_attention_with_mode(
            (queries, keys, values),
            scale,
            window,
            c"".as_ptr(),
            MlxArray::empty_raw(),
            OPERATION,
        )
    }

    /// Applies causal attention restricted to a sliding window: each query
    /// sees itself and the `window_size - 1` keys before it.
    ///
    /// # Errors
    /// Fails when `window_size` is zero or larger than `i32::MAX`, and for the
    /// argument errors [`Self::scaled_dot_product_attention`] reports.
    pub fn causal_windowed_scaled_dot_product_attention(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        scale: f32,
        window_size: u32,
    ) -> Result<MlxArray, MlxCError> {
        const OPERATION: &str = "apply causal windowed scaled dot-product attention";
        validate_attention_arguments(queries, keys, values, scale)?;
        let window = AttentionWindow::new(window_size, OPERATION)?;
        self.windowed_attention_with_mode(
            (queries, keys, values),
            scale,
            window,
            c"causal".as_ptr(),
            MlxArray::empty_raw(),
            OPERATION,
        )
    }

    /// Applies windowed attention that also honors a broadcast-compatible
    /// boolean or additive mask: a key must be inside the window and kept by
    /// the mask.
    ///
    /// # Errors
    /// Fails when `window_size` is zero or larger than `i32::MAX`, and for the
    /// argument errors [`Self::masked_scaled_dot_product_attention`] reports.
    pub fn masked_windowed_scaled_dot_product_attention(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        scale: f32,
        window_size: u32,
        mask: &MlxArray,
    ) -> Result<MlxArray, MlxCError> {
        const OPERATION: &str = "apply masked windowed scaled dot-product attention";
        validate_masked_attention_arguments(queries, keys, values, scale, mask)?;
        let window = AttentionWindow::new(window_size, OPERATION)?;
        self.windowed_attention_with_mode(
            (queries, keys, values),
            scale,
            window,
            c"array".as_ptr(),
            mask.raw(),
            OPERATION,
        )
    }

    // The three attention operands travel as one tuple: they are always
    // passed together, and grouping them keeps the full native contract
    // visible without the argument-count allowance the unwindowed helper
    // needs.
    fn windowed_attention_with_mode(
        &self,
        (queries, keys, values): (&MlxArray, &MlxArray, &MlxArray),
        scale: f32,
        window: AttentionWindow,
        mask_mode: *const std::ffi::c_char,
        mask_array: raw::mlx_array,
        operation: &'static str,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array(operation, |output, stream| {
            // SAFETY: Inputs and stream are live, mode is a static C string,
            // mask/sinks are valid MLX handles, the window is a validated
            // positive int, and output is uniquely writable.
            unsafe {
                raw::mlxrs_fast_scaled_dot_product_attention_windowed(
                    output,
                    queries.raw(),
                    keys.raw(),
                    values.raw(),
                    scale,
                    window.0,
                    mask_mode,
                    mask_array,
                    MlxArray::empty_raw(),
                    // The same eager reference path the unwindowed entries use.
                    false,
                    stream,
                )
            }
        })
    }

    // Keeping the complete native attention contract explicit makes mask-mode mistakes visible.
    #[allow(clippy::too_many_arguments)]
    fn scaled_dot_product_attention_with_mode(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        scale: f32,
        mask_mode: *const std::ffi::c_char,
        mask_array: raw::mlx_array,
        operation: &'static str,
    ) -> Result<MlxArray, MlxCError> {
        self.output_array(operation, |output, stream| {
            // SAFETY: Inputs and stream are live, mode is a static C string,
            // mask/sinks are valid MLX handles, and output is uniquely writable.
            unsafe {
                raw::mlx_fast_scaled_dot_product_attention(
                    output,
                    queries.raw(),
                    keys.raw(),
                    values.raw(),
                    scale,
                    mask_mode,
                    mask_array,
                    MlxArray::empty_raw(),
                    // MLX-C exposes the fused-attention selection; keep
                    // the eager path on the reference implementation.
                    false,
                    stream,
                )
            }
        })
    }
}

/// A sliding window size MLX accepts: positive and representable as a C int.
/// MLX reserves -1 for "no window", which the unwindowed entries already
/// mean, so the windowed entries never pass it.
#[derive(Clone, Copy)]
struct AttentionWindow(i32);

impl AttentionWindow {
    fn new(window_size: u32, operation: &'static str) -> Result<Self, MlxCError> {
        match i32::try_from(window_size) {
            Ok(positive_window) if positive_window > 0 => Ok(Self(positive_window)),
            Ok(_) => Err(runtime_operation_error(
                operation,
                "attention window size must be positive",
            )),
            Err(_) => Err(runtime_operation_error(
                operation,
                "attention window size exceeds the C API integer range",
            )),
        }
    }
}

fn validate_attention_arguments(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
) -> Result<(), MlxCError> {
    const OPERATION: &str = "apply scaled dot-product attention";
    if !scale.is_finite() {
        return Err(runtime_operation_error(
            OPERATION,
            "attention scale must be finite",
        ));
    }
    let query_shape = queries.shape();
    let key_shape = keys.shape();
    let value_shape = values.shape();
    if query_shape.len() != 4 || key_shape.len() != 4 || value_shape.len() != 4 {
        return Err(runtime_operation_error(
            OPERATION,
            "attention query, key, and value tensors must have rank four",
        ));
    }
    // Validate positive geometry before divisibility. In particular, modulo by
    // a zero key/value head count would panic in Rust instead of returning the
    // typed runtime error required at this native boundary.
    if query_shape.iter().any(|dimension| *dimension <= 0)
        || key_shape.iter().any(|dimension| *dimension <= 0)
        || value_shape.iter().any(|dimension| *dimension <= 0)
    {
        return Err(runtime_operation_error(
            OPERATION,
            "attention dimensions must be positive",
        ));
    }
    if key_shape != value_shape {
        return Err(runtime_operation_error(
            OPERATION,
            "attention keys and values must have identical shape",
        ));
    }
    if query_shape[0] != key_shape[0] {
        return Err(runtime_operation_error(
            OPERATION,
            "attention batch dimensions must match",
        ));
    }
    if query_shape[3] != key_shape[3] {
        return Err(runtime_operation_error(
            OPERATION,
            "attention query and key head widths must match",
        ));
    }
    if query_shape[1] % key_shape[1] != 0 {
        return Err(runtime_operation_error(
            OPERATION,
            "attention query heads must be a multiple of key/value heads",
        ));
    }
    Ok(())
}

/// Validates the explicit mask against the query and key sequence dimensions.
fn validate_masked_attention_arguments(
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
    mask: &MlxArray,
) -> Result<(), MlxCError> {
    const OPERATION: &str = "apply masked scaled dot-product attention";
    validate_attention_arguments(queries, keys, values, scale)?;
    let mask_shape = mask.shape();
    if mask_shape.len() < 2 || mask_shape.len() > 4 {
        return Err(runtime_operation_error(
            OPERATION,
            "attention mask must have rank 2, 3, or 4",
        ));
    }
    let query_shape = queries.shape();
    let key_shape = keys.shape();
    let mask_rank = mask_shape.len();
    let mask_query_length = mask_shape[mask_rank - 2];
    let mask_key_length = mask_shape[mask_rank - 1];
    if mask_query_length == 0 || mask_key_length == 0 {
        return Err(runtime_operation_error(
            OPERATION,
            "attention mask sequence dimensions must be positive",
        ));
    }
    if mask_query_length != 1 && mask_query_length != query_shape[2] {
        return Err(runtime_operation_error(
            OPERATION,
            "attention mask query dimension must broadcast to the query length",
        ));
    }
    if mask_key_length != 1 && mask_key_length != key_shape[2] {
        return Err(runtime_operation_error(
            OPERATION,
            "attention mask key dimension must broadcast to the key/value length",
        ));
    }
    Ok(())
}

fn runtime_operation_error(operation: &'static str, description: &'static str) -> MlxCError {
    MlxCError {
        operation,
        description: description.to_owned(),
    }
}
