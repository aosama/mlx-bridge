//! MLX's fast primitives without a family of their own: the gated-delta update
//! and fused cross entropy.

use crate::error::check_status;
use crate::raw;
use crate::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError};

impl MlxBindingsContext {
    /// Applies MLX-C `mlx_fast_gated_delta_update`, returning the outputs first
    /// and the final recurrent state second.
    ///
    /// `queries` and `keys` are shaped `(batch, steps, key_heads, key_dims)`,
    /// `values` is `(batch, steps, value_heads, value_dims)`, `gates` and `beta`
    /// are `(batch, steps, value_heads)`, and `initial_state` is
    /// `(batch, value_heads, value_dims, key_dims)`, zero when absent. `mask` is
    /// a boolean `(batch, steps)` selecting the steps that update the state.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the graph cannot be built.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_update(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        gates: &MlxArray,
        beta: &MlxArray,
        initial_state: Option<&MlxArray>,
        mask: Option<&MlxArray>,
    ) -> Result<MlxArrayVector, MlxCError> {
        let operation: &'static str = "apply the MLX gated-delta update";
        let mut outputs = MlxArrayVector::empty(operation)?;
        // SAFETY: Every array handle and the stream are live, absent optional
        // arrays use MLX's empty handle convention, and the output is uniquely
        // writable.
        let status = unsafe {
            raw::mlx_fast_gated_delta_update(
                outputs.raw_mut(),
                queries.raw(),
                keys.raw(),
                values.raw(),
                gates.raw(),
                beta.raw(),
                initial_state.map_or_else(MlxArray::empty_raw, MlxArray::raw),
                mask.map_or_else(MlxArray::empty_raw, MlxArray::raw),
                self.stream().raw(),
            )
        };
        check_status(status, operation)?;
        Ok(outputs)
    }

    /// Applies MLX-C `mlx_fast_cross_entropy` over `logits` and class `targets`.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the graph cannot be built.
    pub fn fast_cross_entropy(
        &self,
        logits: &MlxArray,
        targets: &MlxArray,
    ) -> Result<MlxArray, MlxCError> {
        let operation: &'static str = "apply MLX fast cross entropy";
        let mut output = MlxArray::empty();
        // SAFETY: Every array handle and the stream are live and the output is
        // uniquely writable.
        let status = unsafe {
            raw::mlx_fast_cross_entropy(
                output.raw_mut(),
                logits.raw(),
                targets.raw(),
                self.stream().raw(),
            )
        };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }
}
