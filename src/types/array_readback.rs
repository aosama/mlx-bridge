//! Whole-array host reads: every path copying an array's elements into Rust
//! memory, with the element type and storage layout checks they all share.

use std::ffi::c_int;

use crate::error::check_status;
use crate::raw;
use crate::types::stream::MlxStream;
use crate::{MlxArray, MlxCError, MlxDtype};

// The bindgen allowlist selects `mlx_*` names, and MLX-C's contiguity queries
// carry a leading underscore, so this one is declared directly against the
// already-linked library, matching the Metal capture declarations.
unsafe extern "C" {
    fn _mlx_array_is_row_contiguous(res: *mut bool, arr: raw::mlx_array) -> c_int;
}

impl MlxArray {
    /// Evaluates the array and copies its float32 elements into Rust-owned
    /// memory in row-major order. Views such as transposes, reversals, and
    /// strided slices read in their logical order.
    ///
    /// # Errors
    /// Fails when the array is not float32 or evaluation fails.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>, MlxCError> {
        self.copy_host_values(
            MlxDtype::Float32,
            raw::mlx_array_data_float32,
            "copy an MLX float32 array",
        )
    }

    /// Evaluates the array and copies its uint32 elements into Rust-owned
    /// memory in row-major order.
    ///
    /// # Errors
    /// Fails when the array is not uint32 or evaluation fails.
    pub fn to_vec_u32(&self) -> Result<Vec<u32>, MlxCError> {
        self.copy_host_values(
            MlxDtype::UInt32,
            raw::mlx_array_data_uint32,
            "copy an MLX uint32 array",
        )
    }

    /// Evaluates the array and copies its uint8 elements into Rust-owned
    /// memory in row-major order.
    ///
    /// # Errors
    /// Fails when the array is not uint8 or evaluation fails.
    pub fn to_vec_u8(&self) -> Result<Vec<u8>, MlxCError> {
        self.copy_host_values(
            MlxDtype::UInt8,
            raw::mlx_array_data_uint8,
            "copy an MLX uint8 array",
        )
    }

    /// Copies a contiguous uint8 array in one host transfer.
    ///
    /// Evaluate first so any wait is attributed to that call; an unevaluated
    /// array is evaluated here rather than read as absent storage.
    ///
    /// # Errors
    /// Fails when the array is not uint8, or when its storage is a view: this
    /// method exists to attribute one direct transfer, so it never copies
    /// through a materialization the way `to_vec_u8` does.
    pub fn copy_evaluated_u8_values(&self) -> Result<Vec<u8>, MlxCError> {
        self.copy_direct_transfer_values(
            MlxDtype::UInt8,
            raw::mlx_array_data_uint8,
            "copy an evaluated MLX uint8 array",
        )
    }

    /// Copies a contiguous uint32 array in one host transfer.
    ///
    /// This split exists so performance attribution can distinguish the
    /// synchronization wait from the small host copy: evaluate first so any
    /// wait is attributed to that call; an unevaluated array is evaluated here
    /// rather than read as absent storage.
    ///
    /// # Errors
    /// Fails when the array is not uint32, or when its storage is a view: this
    /// method never copies through a materialization the way `to_vec_u32` does.
    pub fn copy_evaluated_u32_values(&self) -> Result<Vec<u32>, MlxCError> {
        self.copy_direct_transfer_values(
            MlxDtype::UInt32,
            raw::mlx_array_data_uint32,
            "copy an evaluated MLX uint32 array",
        )
    }

    /// Evaluates the array and copies its complex elements, in row-major
    /// order, as `(real, imaginary)` pairs.
    ///
    /// # Errors
    /// Fails when the array is not complex or evaluation fails.
    pub fn to_vec_complex(&self) -> Result<Vec<(f32, f32)>, MlxCError> {
        let complex_values = self.copy_host_values(
            MlxDtype::Complex64,
            raw::mlx_array_data_complex64,
            "copy an MLX complex array",
        )?;
        let mut value_pairs = Vec::with_capacity(complex_values.len());
        for complex_value in complex_values {
            value_pairs.push((complex_value.re, complex_value.im));
        }
        Ok(value_pairs)
    }

    /// Copies the evaluated boolean storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_bool(&self) -> Result<Vec<bool>, MlxCError> {
        self.copy_host_values(
            MlxDtype::Bool,
            raw::mlx_array_data_bool,
            "copy an MLX boolean array",
        )
    }

    /// Copies the evaluated uint16 storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_u16(&self) -> Result<Vec<u16>, MlxCError> {
        self.copy_host_values(
            MlxDtype::UInt16,
            raw::mlx_array_data_uint16,
            "copy an MLX uint16 array",
        )
    }

    /// Copies the evaluated uint64 storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_u64(&self) -> Result<Vec<u64>, MlxCError> {
        self.copy_host_values(
            MlxDtype::UInt64,
            raw::mlx_array_data_uint64,
            "copy an MLX uint64 array",
        )
    }

    /// Copies the evaluated int8 storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_i8(&self) -> Result<Vec<i8>, MlxCError> {
        self.copy_host_values(
            MlxDtype::Int8,
            raw::mlx_array_data_int8,
            "copy an MLX int8 array",
        )
    }

    /// Copies the evaluated int16 storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_i16(&self) -> Result<Vec<i16>, MlxCError> {
        self.copy_host_values(
            MlxDtype::Int16,
            raw::mlx_array_data_int16,
            "copy an MLX int16 array",
        )
    }

    /// Copies the evaluated int32 storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_i32(&self) -> Result<Vec<i32>, MlxCError> {
        self.copy_host_values(
            MlxDtype::Int32,
            raw::mlx_array_data_int32,
            "copy an MLX int32 array",
        )
    }

    /// Copies the evaluated int64 storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_i64(&self) -> Result<Vec<i64>, MlxCError> {
        self.copy_host_values(
            MlxDtype::Int64,
            raw::mlx_array_data_int64,
            "copy an MLX int64 array",
        )
    }

    /// Copies the evaluated float64 storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_f64(&self) -> Result<Vec<f64>, MlxCError> {
        self.copy_host_values(
            MlxDtype::Float64,
            raw::mlx_array_data_float64,
            "copy an MLX float64 array",
        )
    }

    /// Copies the evaluated float16 storage into Rust memory as raw bits.
    ///
    /// Only on 64-bit ARM, mirroring the float16 guard in the vendored
    /// `mlx/c/half.h`.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    #[cfg(target_arch = "aarch64")]
    pub fn to_vec_f16_bits(&self) -> Result<Vec<u16>, MlxCError> {
        let half_values = self.copy_host_values(
            MlxDtype::Float16,
            raw::mlx_array_data_float16,
            "copy an MLX float16 array",
        )?;
        let mut half_bits = Vec::with_capacity(half_values.len());
        for half_value in half_values {
            half_bits.push(half_value.0);
        }
        Ok(half_bits)
    }

    /// Copies the evaluated bfloat16 storage into Rust memory as raw bits.
    ///
    /// Only on 64-bit ARM, mirroring the bfloat16 guard in the vendored
    /// `mlx/c/half.h`.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    #[cfg(target_arch = "aarch64")]
    pub fn to_vec_bf16_bits(&self) -> Result<Vec<u16>, MlxCError> {
        self.copy_host_values(
            MlxDtype::BFloat16,
            raw::mlx_array_data_bfloat16,
            "copy an MLX bfloat16 array",
        )
    }

    /// Evaluates the array and copies its logical elements, in row-major
    /// order, into Rust memory.
    ///
    /// MLX's data accessors reinterpret the buffer as whatever type was asked
    /// for and ignore strides, so two checks stand between them and undefined
    /// behavior: a float64 read of float32 storage would run past the
    /// allocation, and a forward read from a reversed view's data pointer
    /// (which addresses the view's first logical element, the buffer's last)
    /// would too. Contiguous arrays copy directly; only views pay for a copy.
    pub(crate) fn copy_host_values<T: Clone>(
        &self,
        expected_dtype: MlxDtype,
        data_pointer: unsafe extern "C" fn(raw::mlx_array) -> *const T,
        operation: &'static str,
    ) -> Result<Vec<T>, MlxCError> {
        self.require_dtype(expected_dtype, operation)?;
        self.evaluate()?;
        if self.element_count() == 0 {
            return Ok(Vec::new());
        }
        if self.is_row_contiguous()? {
            return copy_contiguous_values(self, data_pointer, operation);
        }
        let materialized = self.row_major_copy(operation)?;
        copy_contiguous_values(&materialized, data_pointer, operation)
    }

    /// Copies this array's elements only if that takes no materialization.
    pub(crate) fn copy_direct_transfer_values<T: Clone>(
        &self,
        expected_dtype: MlxDtype,
        data_pointer: unsafe extern "C" fn(raw::mlx_array) -> *const T,
        operation: &'static str,
    ) -> Result<Vec<T>, MlxCError> {
        self.require_dtype(expected_dtype, operation)?;
        if self.element_count() == 0 {
            return Ok(Vec::new());
        }
        if !self.is_row_contiguous()? {
            return Err(MlxCError {
                operation,
                description: "array storage is a non-contiguous view; read it through the \
                              materializing to_vec family"
                    .to_owned(),
            });
        }
        copy_contiguous_values(self, data_pointer, operation)
    }

    /// Whether this array's evaluated storage is one row-major contiguous
    /// run, which is what raw host reads require. Transposes, reversals, and
    /// strided slices are views over their parent's buffer and report `false`.
    ///
    /// Evaluates the array first because MLX assigns storage flags only when
    /// the data materializes.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when evaluation or the query
    /// fails.
    pub fn is_row_contiguous(&self) -> Result<bool, MlxCError> {
        self.evaluate()?;
        let mut is_contiguous = false;
        // SAFETY: The output pointer is valid writable storage and `self`
        // owns a live array handle.
        let status = unsafe { _mlx_array_is_row_contiguous(&mut is_contiguous, self.raw_array) };
        check_status(status, "query MLX array contiguity")?;
        Ok(is_contiguous)
    }

    /// Materializes this view into an evaluated row-major array.
    ///
    /// Runs on the CPU stream because the array has no context to name a
    /// stream and the GPU rejects float64, while the CPU copies every dtype
    /// over the same unified memory.
    fn row_major_copy(&self, operation: &'static str) -> Result<Self, MlxCError> {
        let cpu_stream = MlxStream::default_cpu()?;
        let mut materialized = Self::empty();
        // SAFETY: The input and stream handles are live and the output handle
        // is uniquely writable.
        let status = unsafe {
            raw::mlx_contiguous(
                materialized.raw_mut(),
                self.raw_array,
                false,
                cpu_stream.raw(),
            )
        };
        check_status(status, operation)?;
        materialized.require_populated(operation)?;
        materialized.evaluate()?;
        Ok(materialized)
    }
}

/// Copies an evaluated, row-contiguous, dtype-checked array's elements.
fn copy_contiguous_values<T: Clone>(
    array: &MlxArray,
    data_pointer: unsafe extern "C" fn(raw::mlx_array) -> *const T,
    operation: &'static str,
) -> Result<Vec<T>, MlxCError> {
    // SAFETY: The caller established a live, evaluated array of this element
    // type; a null return is rejected below rather than dereferenced.
    let values_pointer = unsafe { data_pointer(array.raw()) };
    if values_pointer.is_null() {
        return Err(MlxCError {
            operation,
            description: "MLX returned a null data pointer after evaluation".to_owned(),
        });
    }
    // SAFETY: Row-contiguous storage of this dtype holds `element_count`
    // consecutive values from the data pointer; they are copied before the
    // borrow ends.
    Ok(unsafe { std::slice::from_raw_parts(values_pointer, array.element_count()) }.to_vec())
}
