//! The MLX array accessors: scalar constructors and setters, per-dtype
//! element and storage readers, contiguity queries, and managed buffers.

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
    /// Creates a boolean scalar array.
    pub fn from_bool(value: bool) -> Self {
        // SAFETY: The scalar constructor takes a plain value and the returned
        // handle enters RAII ownership.
        let raw_array = unsafe { raw::mlx_array_new_bool(value) };
        Self::own(raw_array)
    }

    /// Creates an integer scalar array.
    pub fn from_scalar_i32(value: i32) -> Self {
        // SAFETY: The scalar constructor takes a plain value and the returned
        // handle enters RAII ownership.
        let raw_array = unsafe { raw::mlx_array_new_int(value) };
        Self::own(raw_array)
    }

    /// Creates a float32 scalar array.
    pub fn from_scalar_f32(value: f32) -> Self {
        // SAFETY: The scalar constructor takes a plain value and the returned
        // handle enters RAII ownership.
        let raw_array = unsafe { raw::mlx_array_new_float32(value) };
        Self::own(raw_array)
    }

    /// Creates a float64 scalar array.
    pub fn from_scalar_f64(value: f64) -> Self {
        // SAFETY: The scalar constructor takes a plain value and the returned
        // handle enters RAII ownership.
        let raw_array = unsafe { raw::mlx_array_new_float64(value) };
        Self::own(raw_array)
    }

    /// Creates a complex scalar array from real and imaginary parts.
    pub fn from_scalar_complex(real: f32, imag: f32) -> Self {
        // SAFETY: The scalar constructor takes plain values and the returned
        // handle enters RAII ownership.
        let raw_array = unsafe { raw::mlx_array_new_complex(real, imag) };
        Self::own(raw_array)
    }

    fn own(raw_array: raw::mlx_array) -> Self {
        Self { raw_array }
    }

    /// Creates an array that borrows host memory without copying.
    ///
    /// # Safety
    /// The buffer must stay valid and unmodified until the destructor runs,
    /// its contents must match `dtype`'s element layout, and its element
    /// count must match the shape product.
    pub unsafe fn from_managed_buffer(
        buffer: Vec<u8>,
        shape: &[i32],
        dtype: MlxDtype,
    ) -> Result<Self, MlxCError> {
        let operation: &'static str = "create an MLX array from a managed buffer";
        let element_count = shape.iter().try_fold(1_usize, |product, dimension| {
            product.checked_mul(usize::try_from(*dimension).ok()?)
        });
        if element_count != Some(buffer.len() / Self::byte_size_of_dtype(dtype)) {
            return Err(MlxCError {
                operation,
                description: "buffer byte count does not match the shape and dtype".to_owned(),
            });
        }
        let managed = Box::new(ManagedBuffer { storage: buffer });
        let data_pointer = managed.storage.as_ptr();
        let payload = Box::into_raw(managed);
        let rank = i32::try_from(shape.len()).map_err(|_| MlxCError {
            operation,
            description: "array rank exceeds the C API integer range".to_owned(),
        })?;
        // SAFETY: The boxed buffer stays alive until the payload destructor
        // runs and the destructor reclaims exactly this allocation; the shape
        // slice remains valid for the copying constructor.
        let raw_array = unsafe {
            raw::mlx_array_new_data_managed_payload(
                data_pointer.cast_mut().cast(),
                shape.as_ptr(),
                rank,
                dtype.to_raw(),
                payload.cast(),
                Some(drop_managed_payload),
            )
        };
        let array = Self::own(raw_array);
        array.require_populated(operation)?;
        Ok(array)
    }

    /// Replaces this handle's value with a boolean scalar.
    pub fn set_bool(&mut self, value: bool) -> Result<(), MlxCError> {
        // SAFETY: The handle is live and the value is plain.
        let status = unsafe { raw::mlx_array_set_bool(self.raw_mut(), value) };
        check_status(status, "replace an MLX array value")
    }

    /// Replaces this handle's value with an integer scalar.
    pub fn set_i32(&mut self, value: i32) -> Result<(), MlxCError> {
        // SAFETY: The handle is live and the value is plain.
        let status = unsafe { raw::mlx_array_set_int(self.raw_mut(), value) };
        check_status(status, "replace an MLX array value")
    }

    /// Replaces this handle's value with a float32 scalar.
    pub fn set_f32(&mut self, value: f32) -> Result<(), MlxCError> {
        // SAFETY: The handle is live and the value is plain.
        let status = unsafe { raw::mlx_array_set_float32(self.raw_mut(), value) };
        check_status(status, "replace an MLX array value")
    }

    /// Replaces this handle's value with a float64 scalar.
    pub fn set_f64(&mut self, value: f64) -> Result<(), MlxCError> {
        // SAFETY: The handle is live and the value is plain.
        let status = unsafe { raw::mlx_array_set_float64(self.raw_mut(), value) };
        check_status(status, "replace an MLX array value")
    }

    /// Replaces this handle's value with a complex scalar.
    pub fn set_complex(&mut self, real: f32, imag: f32) -> Result<(), MlxCError> {
        // SAFETY: The handle is live and the values are plain.
        let status = unsafe { raw::mlx_array_set_complex(self.raw_mut(), real, imag) };
        check_status(status, "replace an MLX array value")
    }

    /// The per-element byte size of the array's dtype.
    #[must_use]
    pub fn item_size(&self) -> usize {
        // SAFETY: `self` owns a live MLX array handle.
        unsafe { raw::mlx_array_itemsize(self.raw_array) }
    }

    /// The element strides in elements per dimension.
    ///
    /// MLX-C publishes these as unsigned values, so a reversed view's stride
    /// of -1 reads as `usize::MAX`. Use `is_row_contiguous` to decide whether
    /// storage can be read directly.
    #[must_use]
    pub fn strides(&self) -> Vec<usize> {
        // SAFETY: `self` owns a live MLX array handle.
        let dimension_count = unsafe { raw::mlx_array_ndim(self.raw_array) };
        if dimension_count == 0 {
            return Vec::new();
        }
        // SAFETY: MLX keeps this stride storage alive with the array and it
        // addresses exactly `dimension_count` entries.
        let stride_pointer = unsafe { raw::mlx_array_strides(self.raw_array) };
        if stride_pointer.is_null() {
            return Vec::new();
        }
        // SAFETY: The preceding MLX contract establishes the pointer length;
        // the values are copied before returning.
        unsafe { std::slice::from_raw_parts(stride_pointer, dimension_count) }.to_vec()
    }

    /// The size of one dimension; a negative axis counts from the last.
    ///
    /// MLX-C reports an out-of-range axis as 0, indistinguishable from a
    /// zero-sized dimension. Compare against `shape().len()` when the axis is
    /// not already known to be valid.
    #[must_use]
    pub fn dimension(&self, axis: i32) -> i32 {
        // SAFETY: `self` owns a live MLX array handle.
        unsafe { raw::mlx_array_dim(self.raw_array, axis) }
    }

    /// The dtype's element byte size.
    #[must_use]
    pub fn byte_size_of_dtype(dtype: MlxDtype) -> usize {
        // SAFETY: The dtype query is a pure function of the tag.
        unsafe { raw::mlx_dtype_size(dtype.to_raw()) }
    }

    /// Copies the boolean scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_bool(&self) -> Result<bool, MlxCError> {
        self.require_dtype(MlxDtype::Bool, "read an MLX boolean scalar")?;
        let mut value = false;
        // SAFETY: The output pointer is valid writable storage and the guard
        // above established this live array's element type.
        let status = unsafe { raw::mlx_array_item_bool(&mut value, self.raw_array) };
        check_status(status, "read an MLX boolean scalar")?;
        Ok(value)
    }

    /// Copies the uint8 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_u8(&self) -> Result<u8, MlxCError> {
        read_scalar(
            self,
            MlxDtype::UInt8,
            raw::mlx_array_item_uint8,
            "read an MLX uint8 scalar",
        )
    }

    /// Copies the uint16 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_u16(&self) -> Result<u16, MlxCError> {
        read_scalar(
            self,
            MlxDtype::UInt16,
            raw::mlx_array_item_uint16,
            "read an MLX uint16 scalar",
        )
    }

    /// Copies the uint64 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_u64(&self) -> Result<u64, MlxCError> {
        read_scalar(
            self,
            MlxDtype::UInt64,
            raw::mlx_array_item_uint64,
            "read an MLX uint64 scalar",
        )
    }

    /// Copies the int8 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_i8(&self) -> Result<i8, MlxCError> {
        read_scalar(
            self,
            MlxDtype::Int8,
            raw::mlx_array_item_int8,
            "read an MLX int8 scalar",
        )
    }

    /// Copies the int16 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_i16(&self) -> Result<i16, MlxCError> {
        read_scalar(
            self,
            MlxDtype::Int16,
            raw::mlx_array_item_int16,
            "read an MLX int16 scalar",
        )
    }

    /// Copies the int32 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_i32(&self) -> Result<i32, MlxCError> {
        read_scalar(
            self,
            MlxDtype::Int32,
            raw::mlx_array_item_int32,
            "read an MLX int32 scalar",
        )
    }

    /// Copies the int64 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_i64(&self) -> Result<i64, MlxCError> {
        read_scalar(
            self,
            MlxDtype::Int64,
            raw::mlx_array_item_int64,
            "read an MLX int64 scalar",
        )
    }

    /// Copies the float32 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_f32(&self) -> Result<f32, MlxCError> {
        read_scalar(
            self,
            MlxDtype::Float32,
            raw::mlx_array_item_float32,
            "read an MLX float32 scalar",
        )
    }

    /// Copies the float64 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_f64(&self) -> Result<f64, MlxCError> {
        read_scalar(
            self,
            MlxDtype::Float64,
            raw::mlx_array_item_float64,
            "read an MLX float64 scalar",
        )
    }

    /// Copies the complex scalar value as real and imaginary parts.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_complex(&self) -> Result<(f32, f32), MlxCError> {
        self.require_dtype(MlxDtype::Complex64, "read an MLX complex scalar")?;
        let mut value = raw::mlx_complex64_t { re: 0.0, im: 0.0 };
        // SAFETY: The output pointer is valid writable storage and the guard
        // above established this live array's element type.
        let status = unsafe { raw::mlx_array_item_complex64(&mut value, self.raw_array) };
        check_status(status, "read an MLX complex scalar")?;
        Ok((value.re, value.im))
    }

    /// Copies the float16 scalar value as raw bits.
    ///
    /// Only on 64-bit ARM: the vendored `mlx/c/half.h` declares the float16
    /// entry points under `defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) ||
    /// defined(__aarch64__)`, so the bridge mirrors that availability.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    #[cfg(target_arch = "aarch64")]
    pub fn item_f16_bits(&self) -> Result<u16, MlxCError> {
        self.require_dtype(MlxDtype::Float16, "read an MLX float16 scalar")?;
        let mut value = raw::__BindgenFloat16(0);
        // SAFETY: The output pointer is valid writable storage and the guard
        // above established this live array's element type.
        let status = unsafe { raw::mlx_array_item_float16(&mut value, self.raw_array) };
        check_status(status, "read an MLX float16 scalar")?;
        Ok(value.0)
    }

    /// Copies the bfloat16 scalar value as raw bits.
    ///
    /// Only on 64-bit ARM, mirroring the `__bf16` guard in the vendored
    /// `mlx/c/half.h`.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    #[cfg(target_arch = "aarch64")]
    pub fn item_bf16_bits(&self) -> Result<u16, MlxCError> {
        self.require_dtype(MlxDtype::BFloat16, "read an MLX bfloat16 scalar")?;
        let mut value: u16 = 0;
        // SAFETY: The output pointer is valid writable storage and the guard
        // above established this live array's element type.
        let status = unsafe { raw::mlx_array_item_bfloat16(&mut value, self.raw_array) };
        check_status(status, "read an MLX bfloat16 scalar")?;
        Ok(value)
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

    /// Fails unless this populated array holds exactly `expected_dtype`.
    pub(crate) fn require_dtype(
        &self,
        expected_dtype: MlxDtype,
        operation: &'static str,
    ) -> Result<(), MlxCError> {
        if self.is_empty() {
            return Err(MlxCError {
                operation,
                description: "array handle is empty".to_owned(),
            });
        }
        let actual_dtype = self.dtype();
        if actual_dtype != expected_dtype {
            return Err(MlxCError {
                operation,
                description: format!("array dtype is {actual_dtype:?}, not {expected_dtype:?}"),
            });
        }
        Ok(())
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

/// Reads one scalar after confirming the element type, because MLX's
/// `item<T>()` reinterprets the buffer rather than converting it: a float64
/// read of a float32 scalar would run past the four-byte allocation.
fn read_scalar<T>(
    array: &MlxArray,
    expected_dtype: MlxDtype,
    reader: unsafe extern "C" fn(*mut T, raw::mlx_array) -> i32,
    operation: &'static str,
) -> Result<T, MlxCError> {
    array.require_dtype(expected_dtype, operation)?;
    let mut value: T = unsafe { std::mem::zeroed() };
    // SAFETY: The output pointer is valid writable storage and the guard above
    // established this live array's element type.
    let status = unsafe { reader(&mut value, array.raw()) };
    check_status(status, operation)?;
    Ok(value)
}

/// A host buffer whose lifetime MLX owns through the managed-array
/// destructor.
struct ManagedBuffer {
    storage: Vec<u8>,
}

/// The destructor trampoline for payload-managed buffers.
unsafe extern "C" fn drop_managed_payload(payload: *mut std::os::raw::c_void) {
    if payload.is_null() {
        return;
    }
    // SAFETY: The payload was created by `Box::into_raw` of a `ManagedBuffer`
    // and this destructor runs exactly once per managed array.
    drop(unsafe { Box::from_raw(payload.cast::<ManagedBuffer>()) });
}
