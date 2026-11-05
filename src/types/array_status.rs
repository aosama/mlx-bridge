//! Array status queries: evaluation progress, storage layout, and printing.

use std::ffi::c_int;

use crate::error::check_status;
use crate::raw;
use crate::{MlxArray, MlxCError, MlxString};

// The bindgen allowlist selects `mlx_*` names, and these MLX-C queries carry a
// leading underscore, so they are declared directly against the linked library.
unsafe extern "C" {
    fn _mlx_array_is_available(res: *mut bool, arr: raw::mlx_array) -> c_int;
    fn _mlx_array_is_contiguous(res: *mut bool, arr: raw::mlx_array) -> c_int;
    fn _mlx_array_is_col_contiguous(res: *mut bool, arr: raw::mlx_array) -> c_int;
}

impl MlxArray {
    /// Whether the array's value is ready to read without waiting. Unlike the
    /// contiguity queries this never evaluates, so it can watch work started
    /// through `MlxBindingsContext::async_eval_arrays`.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the query fails.
    pub fn is_available(&self) -> Result<bool, MlxCError> {
        let mut is_available = false;
        // SAFETY: The output pointer is valid writable storage and `self`
        // owns a live array handle.
        let status = unsafe { _mlx_array_is_available(&mut is_available, self.raw_array) };
        check_status(status, "query MLX array availability")?;
        Ok(is_available)
    }

    /// Blocks until this array's value is ready, scheduling it first if nothing
    /// has yet, after which `is_available` reports `true`.
    ///
    /// This deliberately goes through evaluation rather than MLX-C's
    /// `_mlx_array_wait`: on an array nothing has scheduled, that call finds no
    /// event to wait on and marks the array available with no data behind it,
    /// leaving every later read of it or anything built on it a null buffer.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when evaluation fails.
    pub fn wait(&self) -> Result<(), MlxCError> {
        self.evaluate()
    }

    /// Whether the evaluated storage is one contiguous run in any axis order.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when evaluation or the query
    /// fails.
    pub fn is_contiguous(&self) -> Result<bool, MlxCError> {
        self.evaluate()?;
        let mut is_contiguous = false;
        // SAFETY: The output pointer is valid writable storage and `self`
        // owns a live array handle.
        let status = unsafe { _mlx_array_is_contiguous(&mut is_contiguous, self.raw_array) };
        check_status(status, "query MLX array contiguity")?;
        Ok(is_contiguous)
    }

    /// Whether the evaluated storage is one column-major contiguous run.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when evaluation or the query
    /// fails.
    pub fn is_col_contiguous(&self) -> Result<bool, MlxCError> {
        self.evaluate()?;
        let mut is_contiguous = false;
        // SAFETY: The output pointer is valid writable storage and `self`
        // owns a live array handle.
        let status = unsafe { _mlx_array_is_col_contiguous(&mut is_contiguous, self.raw_array) };
        check_status(status, "query MLX array column contiguity")?;
        Ok(is_contiguous)
    }

    /// Renders the array as MLX prints it, evaluating it first. Precision and
    /// boolean spelling follow `set_print_format`.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when rendering fails.
    pub fn to_string_lossy(&self) -> Result<String, MlxCError> {
        let operation: &'static str = "describe an MLX array";
        let mut description = MlxString::empty();
        // SAFETY: `self` owns a live array and the output pointer is valid
        // writable storage for one string handle.
        let status = unsafe { raw::mlx_array_tostring(description.raw_mut(), self.raw_array) };
        check_status(status, operation)?;
        description.require_populated(operation)?;
        Ok(description.to_string_lossy())
    }
}
