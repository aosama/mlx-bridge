//! MLX's configuration values: the integer tunables MLX otherwise reads from
//! `MLX_*` environment variables, such as `MLX_MAX_OPS_PER_BUFFER`, set and
//! read in code instead (MLX's `config` module, added in v0.32.5).
//!
//! Lookups resolve in MLX's order: this thread's scopes, newest first, then
//! process values, then the environment variable of the same name, then the
//! caller's default. A value takes effect where MLX reads the name after the
//! change; MLX reads some names once, for example when it first creates a
//! device or stream, so set those before the first computation.
//!
//! These reach MLX through the gap shim, because MLX-C v0.8.0 predates the
//! module. MLX's `register_config_handler`, which replaces every lookup in
//! the process with one C function, is deliberately not wrapped; the
//! reviewed entry in the upstream audit records why.

use std::ffi::{CString, c_char};
use std::marker::PhantomData;

use crate::error::check_status;
use crate::{MlxCError, raw};

/// A configuration name as a C string; names with interior NUL bytes cannot
/// cross the C boundary and fail here instead of being truncated.
fn config_name_argument(name: &str, operation: &'static str) -> Result<CString, MlxCError> {
    CString::new(name).map_err(|_| MlxCError {
        operation,
        description: "configuration name contains an interior null byte".to_owned(),
    })
}

/// Sets a configuration value for the whole process. The value outranks the
/// environment variable of the same name.
///
/// # Errors
/// Fails when `name` contains a NUL byte or MLX rejects the update.
pub fn config_update(name: &str, value: i32) -> Result<(), MlxCError> {
    let operation: &'static str = "update an MLX configuration value";
    let name_argument = config_name_argument(name, operation)?;
    // SAFETY: The name outlives the call; MLX copies it into its own table.
    let status = unsafe { raw::mlxrs_config_update(name_argument.as_ptr(), value) };
    check_status(status, operation)
}

/// Reads a configuration value, or `default_value` when nothing sets it.
///
/// MLX records the answer of the first read of a name, environment or
/// default, as that name's process value, so later reads of the same name
/// return it even with a different default.
///
/// # Errors
/// Fails when `name` contains a NUL byte or MLX rejects the lookup.
pub fn config_get(name: &str, default_value: i32) -> Result<i32, MlxCError> {
    let operation: &'static str = "read an MLX configuration value";
    let name_argument = config_name_argument(name, operation)?;
    let mut configured_value = 0;
    // SAFETY: The name outlives the call and the output pointer is valid
    // writable storage.
    let status = unsafe {
        raw::mlxrs_config_get(&mut configured_value, name_argument.as_ptr(), default_value)
    };
    check_status(status, operation)?;
    Ok(configured_value)
}

/// Configuration values that apply on the creating thread until the scope
/// drops, outranking process values and the environment.
///
/// MLX keeps scopes in a thread-local list, so a scope cannot move to or be
/// dropped on another thread: the type is neither `Send` nor `Sync`.
/// Scopes may be dropped in any order.
///
/// ```compile_fail
/// fn require_send<T: Send>() {}
/// require_send::<mlx_bridge::MlxConfigScope>();
/// ```
///
/// ```compile_fail
/// fn require_sync<T: Sync>() {}
/// require_sync::<mlx_bridge::MlxConfigScope>();
/// ```
#[derive(Debug)]
pub struct MlxConfigScope {
    raw_scope: raw::mlxrs_config_scope,
    // A raw-pointer marker removes Send and Sync: MLX erases the scope from
    // the calling thread's list on drop, and erasing it from another
    // thread's list is undefined behavior.
    thread_bound: PhantomData<*const ()>,
}

impl MlxConfigScope {
    /// Applies `name_value_pairs` on this thread until the scope drops. A
    /// name given more than once takes its last value.
    ///
    /// # Errors
    /// Fails when a name contains a NUL byte or MLX rejects the scope.
    pub fn new(name_value_pairs: &[(&str, i32)]) -> Result<Self, MlxCError> {
        let operation: &'static str = "open an MLX configuration scope";
        let mut owned_names = Vec::with_capacity(name_value_pairs.len());
        let mut scoped_values = Vec::with_capacity(name_value_pairs.len());
        for (name, value) in name_value_pairs {
            owned_names.push(config_name_argument(name, operation)?);
            scoped_values.push(*value);
        }
        let mut name_pointers: Vec<*const c_char> = Vec::with_capacity(owned_names.len());
        for owned_name in &owned_names {
            name_pointers.push(owned_name.as_ptr());
        }
        let mut raw_scope = raw::mlxrs_config_scope {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: The name pointers and values outlive the call, both arrays
        // hold `scoped_values.len()` entries, and the output is writable.
        let status = unsafe {
            raw::mlxrs_config_scope_new(
                &mut raw_scope,
                name_pointers.as_ptr(),
                scoped_values.as_ptr(),
                scoped_values.len(),
            )
        };
        check_status(status, operation)?;
        Ok(Self {
            raw_scope,
            thread_bound: PhantomData,
        })
    }
}

impl Drop for MlxConfigScope {
    fn drop(&mut self) {
        // SAFETY: This owner frees its scope exactly once, on the thread that
        // created it (the type is not Send), and never touches it afterward.
        unsafe {
            raw::mlxrs_config_scope_free(self.raw_scope);
        }
    }
}
