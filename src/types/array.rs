use crate::{
    MlxCError,
    error::{check_status, clear_captured_mlx_error, take_captured_mlx_error},
    raw,
};

/// MLX array element types exposed without leaking generated C declarations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlxDtype {
    Bool,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    BFloat16,
    Complex64,
}

impl MlxDtype {
    pub(crate) fn from_raw(raw_dtype: raw::mlx_dtype) -> Self {
        match raw_dtype {
            raw::mlx_dtype__MLX_BOOL => Self::Bool,
            raw::mlx_dtype__MLX_UINT8 => Self::UInt8,
            raw::mlx_dtype__MLX_UINT16 => Self::UInt16,
            raw::mlx_dtype__MLX_UINT32 => Self::UInt32,
            raw::mlx_dtype__MLX_UINT64 => Self::UInt64,
            raw::mlx_dtype__MLX_INT8 => Self::Int8,
            raw::mlx_dtype__MLX_INT16 => Self::Int16,
            raw::mlx_dtype__MLX_INT32 => Self::Int32,
            raw::mlx_dtype__MLX_INT64 => Self::Int64,
            raw::mlx_dtype__MLX_FLOAT16 => Self::Float16,
            raw::mlx_dtype__MLX_FLOAT32 => Self::Float32,
            raw::mlx_dtype__MLX_FLOAT64 => Self::Float64,
            raw::mlx_dtype__MLX_BFLOAT16 => Self::BFloat16,
            raw::mlx_dtype__MLX_COMPLEX64 => Self::Complex64,
            _ => Self::Bool,
        }
    }

    /// The raw MLX dtype tag for raw-operation builders outside this crate.
    pub const fn to_raw(self) -> raw::mlx_dtype {
        match self {
            Self::Bool => raw::mlx_dtype__MLX_BOOL,
            Self::UInt8 => raw::mlx_dtype__MLX_UINT8,
            Self::UInt16 => raw::mlx_dtype__MLX_UINT16,
            Self::UInt32 => raw::mlx_dtype__MLX_UINT32,
            Self::UInt64 => raw::mlx_dtype__MLX_UINT64,
            Self::Int8 => raw::mlx_dtype__MLX_INT8,
            Self::Int16 => raw::mlx_dtype__MLX_INT16,
            Self::Int32 => raw::mlx_dtype__MLX_INT32,
            Self::Int64 => raw::mlx_dtype__MLX_INT64,
            Self::Float16 => raw::mlx_dtype__MLX_FLOAT16,
            Self::Float32 => raw::mlx_dtype__MLX_FLOAT32,
            Self::Float64 => raw::mlx_dtype__MLX_FLOAT64,
            Self::BFloat16 => raw::mlx_dtype__MLX_BFLOAT16,
            Self::Complex64 => raw::mlx_dtype__MLX_COMPLEX64,
        }
    }
}

/// Owned MLX array handle released exactly once through the official C API.
#[derive(Debug)]
pub struct MlxArray {
    pub(crate) raw_array: raw::mlx_array,
}

impl MlxArray {
    /// Creates a float32 array from host values laid out in row-major order. The
    /// values are copied, so later changes to the slice do not affect the array.
    ///
    /// # Errors
    /// Fails when the shape's element count differs from the value count.
    pub fn from_f32(values: &[f32], shape: &[i32]) -> Result<Self, MlxCError> {
        Self::from_host_values(
            values,
            shape,
            MlxDtype::Float32,
            "create an MLX float32 array",
        )
    }

    /// Creates an int32 array from host values laid out in row-major order.
    ///
    /// # Errors
    /// Fails when the shape's element count differs from the value count.
    pub fn from_i32(values: &[i32], shape: &[i32]) -> Result<Self, MlxCError> {
        Self::from_host_values(values, shape, MlxDtype::Int32, "create an MLX int32 array")
    }

    /// Creates a uint32 array from host values laid out in row-major order.
    ///
    /// # Errors
    /// Fails when the shape's element count differs from the value count.
    pub fn from_u32(values: &[u32], shape: &[i32]) -> Result<Self, MlxCError> {
        Self::from_host_values(
            values,
            shape,
            MlxDtype::UInt32,
            "create an MLX uint32 array",
        )
    }

    /// The one path host values take into MLX. Callers pair `T` with the
    /// element type whose storage layout it matches.
    pub(crate) fn from_host_values<T>(
        values: &[T],
        shape: &[i32],
        dtype: MlxDtype,
        operation: &'static str,
    ) -> Result<Self, MlxCError> {
        validate_shape(values.len(), shape)?;
        let Ok(dimension_count) = i32::try_from(shape.len()) else {
            return Err(MlxCError {
                operation,
                description: "array rank exceeds the C API integer range".to_owned(),
            });
        };
        // SAFETY: The slices remain valid for this copying constructor, their
        // lengths were validated against the shape, the caller's `T` matches
        // `dtype`'s layout, and the returned handle enters RAII ownership.
        let raw_array = unsafe {
            raw::mlx_array_new_data(
                values.as_ptr().cast(),
                shape.as_ptr(),
                dimension_count,
                dtype.to_raw(),
            )
        };
        let array = Self { raw_array };
        array.require_populated(operation)?;
        Ok(array)
    }

    /// Creates an empty handle holding no array, for passing as an output
    /// parameter. Evaluating it or reading its values returns an error.
    pub fn empty() -> Self {
        // SAFETY: The runtime error handler is installed before model loading,
        // and the official API's empty placeholder is immediately placed under
        // RAII ownership before a fallible operation populates it.
        let raw_array = unsafe { raw::mlx_array_new() };
        Self { raw_array }
    }

    /// Whether this handle holds no array at all. A zero-sized array is a real
    /// array, so this reports `false` for it.
    pub fn is_empty(&self) -> bool {
        self.raw_array.ctx.is_null()
    }

    /// The underlying MLX-C handle, still owned by this value, for calling
    /// `raw` functions directly.
    pub const fn raw(&self) -> raw::mlx_array {
        self.raw_array
    }

    /// The official absent-optional-handle placeholder for raw MLX calls.
    ///
    /// Compiled-graph builders outside this crate mirror the runtime's own
    /// operations and need the same empty convention for absent `freqs`,
    /// `sinks`, and `global_scale` handles; the placeholder owns nothing and
    /// is always valid to pass by value.
    pub const fn empty_raw() -> raw::mlx_array {
        raw::mlx_array {
            ctx: std::ptr::null_mut(),
        }
    }

    /// A pointer to the underlying MLX-C handle, for `raw` functions that write
    /// their result through an output parameter.
    pub fn raw_mut(&mut self) -> *mut raw::mlx_array {
        &mut self.raw_array
    }

    /// Validates that a returned handle is populated, translating a parked
    /// payload-closure failure or the captured MLX-C description into a typed
    /// failure.
    ///
    /// This is a pure translation: embedding-specific error classification
    /// (for example an active memory ceiling) is embedding policy and happens
    /// when the consumer converts this error at its boundary.
    pub fn require_populated(&self, operation: &'static str) -> Result<(), MlxCError> {
        if !self.is_empty() {
            clear_captured_mlx_error();
            crate::error::clear_closure_error();
            return Ok(());
        }
        if let Some(failure) = crate::error::take_closure_error() {
            clear_captured_mlx_error();
            return Err(failure);
        }
        let description = take_captured_mlx_error()
            .unwrap_or_else(|| "MLX returned an empty array handle".to_owned());
        Err(MlxCError {
            operation,
            description,
        })
    }

    /// Returns the dimensions of this live array.
    #[must_use]
    pub fn shape(&self) -> Vec<i32> {
        // SAFETY: `self` exclusively owns a live MLX array handle.
        let dimension_count = unsafe { raw::mlx_array_ndim(self.raw_array) };
        if dimension_count == 0 {
            return Vec::new();
        }
        // SAFETY: MLX keeps this shape storage alive with the array. The
        // pointer addresses exactly `dimension_count` dimensions.
        let shape_pointer = unsafe { raw::mlx_array_shape(self.raw_array) };
        if shape_pointer.is_null() {
            return Vec::new();
        }
        // SAFETY: The preceding MLX contract establishes the pointer length;
        // this method copies the values before returning.
        unsafe { std::slice::from_raw_parts(shape_pointer, dimension_count) }.to_vec()
    }

    /// Returns the array's element type.
    #[must_use]
    pub fn dtype(&self) -> MlxDtype {
        // SAFETY: `self` exclusively owns a live MLX array handle.
        MlxDtype::from_raw(unsafe { raw::mlx_array_dtype(self.raw_array) })
    }

    /// Returns the number of logical elements.
    #[must_use]
    pub fn element_count(&self) -> usize {
        // SAFETY: `self` exclusively owns a live MLX array handle.
        unsafe { raw::mlx_array_size(self.raw_array) }
    }

    /// Returns the logical tensor payload size in bytes.
    #[must_use]
    pub fn byte_count(&self) -> usize {
        // SAFETY: `self` exclusively owns a live MLX array handle.
        unsafe { raw::mlx_array_nbytes(self.raw_array) }
    }

    /// Materializes this lazy array and reports runtime failures without exit.
    pub fn evaluate(&self) -> Result<(), MlxCError> {
        // SAFETY: `self` owns a live handle and MLX evaluation does not retain
        // any Rust borrow beyond this call.
        let status = unsafe { raw::mlx_array_eval(self.raw_array) };
        check_status(status, "evaluate an MLX array")
    }

    /// Evaluates a scalar array and copies its value as an unsigned token ID.
    ///
    /// # Errors
    /// Fails when the array is not a uint32 scalar. MLX validates the size but
    /// reinterprets rather than converts the element, so the type is checked
    /// here.
    pub fn item_u32(&self) -> Result<u32, MlxCError> {
        self.require_dtype(MlxDtype::UInt32, "read an MLX uint32 scalar")?;
        self.evaluate()?;
        let mut scalar_value = 0_u32;
        // SAFETY: The output pointer is valid writable storage and `self` owns
        // a live evaluated uint32 array. MLX validates the scalar size.
        let status = unsafe { raw::mlx_array_item_uint32(&mut scalar_value, self.raw_array) };
        check_status(status, "read an MLX uint32 scalar")?;
        Ok(scalar_value)
    }

    /// Creates a shared reference to the same underlying MLX data.
    ///
    /// MLX uses reference counting, so this is cheap and does not copy the
    /// tensor data. The returned array shares the same lazy graph and
    /// evaluated buffers as the source.
    pub fn retain(&self) -> Result<Self, MlxCError> {
        let mut retained_array = Self::empty();
        // SAFETY: `retained_array` owns a live (empty) handle and `self` owns a
        // live source handle. `mlx_array_set` copies the source reference into
        // the destination.
        let status = unsafe { raw::mlx_array_set(retained_array.raw_mut(), self.raw_array) };
        check_status(status, "retain an MLX array")?;
        if retained_array.is_empty() {
            return Err(MlxCError {
                operation: "retain an MLX array",
                description: "MLX returned an empty handle after set".to_owned(),
            });
        }
        Ok(retained_array)
    }
}

fn validate_shape(element_count: usize, shape: &[i32]) -> Result<(), MlxCError> {
    let shaped_element_count = shape.iter().try_fold(1_usize, |product, dimension| {
        let dimension = usize::try_from(*dimension).ok()?;
        product.checked_mul(dimension)
    });
    if shaped_element_count != Some(element_count) {
        return Err(MlxCError {
            operation: "create an MLX array",
            description: "shape element count does not match the provided values".to_owned(),
        });
    }
    Ok(())
}

impl Drop for MlxArray {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live array exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_array_free(self.raw_array);
        }
    }
}
