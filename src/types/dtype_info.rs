//! Element type queries that need MLX's own tables.

use crate::error::check_status;
use crate::raw;
use crate::{MlxArrayVector, MlxCError, MlxDtype};

/// A floating-point element type's limits, as MLX's `finfo` reports them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlxFloatInfo {
    /// The element's width in bits.
    pub bits: i32,
    /// The most negative finite value.
    pub min: f64,
    /// The largest finite value.
    pub max: f64,
    /// The gap between 1.0 and the next representable value.
    pub eps: f64,
    /// The smallest positive normal value.
    pub smallest_normal: f64,
}

impl MlxDtype {
    /// This floating-point type's limits.
    ///
    /// # Errors
    /// Fails for an element type that is not floating point.
    pub fn float_info(self) -> Result<MlxFloatInfo, MlxCError> {
        let mut limits = MlxFloatInfo {
            bits: 0,
            min: 0.0,
            max: 0.0,
            eps: 0.0,
            smallest_normal: 0.0,
        };
        // SAFETY: Every output pointer is valid writable storage and the
        // element type is a plain value.
        let status = unsafe {
            raw::mlxrs_finfo(
                &mut limits.bits,
                &mut limits.min,
                &mut limits.max,
                &mut limits.eps,
                &mut limits.smallest_normal,
                self.to_raw(),
            )
        };
        check_status(status, "read MLX floating-point limits")?;
        Ok(limits)
    }
}

/// The element type the arrays promote to together; `Bool`, promotion's
/// starting point, for an empty vector.
///
/// # Errors
/// Returns the captured MLX-C description when the query fails.
pub fn result_type(arrays: &MlxArrayVector) -> Result<MlxDtype, MlxCError> {
    let mut promoted_dtype = raw::mlx_dtype__MLX_BOOL;
    // SAFETY: The output pointer is valid writable storage and the vector
    // handle is live.
    let status = unsafe { raw::mlxrs_result_type(&mut promoted_dtype, arrays.raw()) };
    check_status(status, "read the MLX result type")?;
    Ok(MlxDtype::from_raw(promoted_dtype))
}
