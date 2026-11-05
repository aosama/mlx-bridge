//! Process-wide array print settings.

use crate::MlxCError;
use crate::error::check_status;
use crate::raw;

/// How arrays render through `MlxArray::to_string_lossy`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxPrintFormat {
    /// Floating-point digits after the point; negative selects MLX's default.
    pub precision: i32,
    /// Prints booleans as `True` and `False` rather than `true` and `false`.
    pub capitalize_bool: bool,
}

/// Replaces the process-wide print format.
///
/// # Errors
/// Returns the captured MLX-C description when the update fails.
pub fn set_print_format(format: MlxPrintFormat) -> Result<(), MlxCError> {
    // SAFETY: Both arguments are plain values.
    let status = unsafe { raw::mlxrs_set_printoptions(format.precision, format.capitalize_bool) };
    check_status(status, "set the MLX print format")
}

/// The process-wide print format.
///
/// # Errors
/// Returns the captured MLX-C description when the read fails.
pub fn print_format() -> Result<MlxPrintFormat, MlxCError> {
    let mut format = MlxPrintFormat {
        precision: 0,
        capitalize_bool: false,
    };
    // SAFETY: Both output pointers are valid writable storage.
    let status = unsafe {
        raw::mlxrs_get_global_formatter(&mut format.precision, &mut format.capitalize_bool)
    };
    check_status(status, "read the MLX print format")?;
    Ok(format)
}
