//! Array constructors for every element type MLX stores, so no element type
//! needs a cast through another one or a detour through the raw layer.

use crate::raw;
use crate::{MlxArray, MlxCError, MlxDtype};

impl MlxArray {
    /// Creates a uint8 array from host values.
    ///
    /// # Errors
    /// Fails when the shape's element count differs from the value count.
    pub fn from_u8(values: &[u8], shape: &[i32]) -> Result<Self, MlxCError> {
        Self::from_host_values(values, shape, MlxDtype::UInt8, "create an MLX uint8 array")
    }

    /// Creates a uint16 array from host values.
    ///
    /// # Errors
    /// Fails when the shape's element count differs from the value count.
    pub fn from_u16(values: &[u16], shape: &[i32]) -> Result<Self, MlxCError> {
        Self::from_host_values(
            values,
            shape,
            MlxDtype::UInt16,
            "create an MLX uint16 array",
        )
    }

    /// Creates a uint64 array from host values.
    ///
    /// # Errors
    /// Fails when the shape's element count differs from the value count.
    pub fn from_u64(values: &[u64], shape: &[i32]) -> Result<Self, MlxCError> {
        Self::from_host_values(
            values,
            shape,
            MlxDtype::UInt64,
            "create an MLX uint64 array",
        )
    }

    /// Creates an int8 array from host values.
    ///
    /// # Errors
    /// Fails when the shape's element count differs from the value count.
    pub fn from_i8(values: &[i8], shape: &[i32]) -> Result<Self, MlxCError> {
        Self::from_host_values(values, shape, MlxDtype::Int8, "create an MLX int8 array")
    }

    /// Creates an int16 array from host values.
    ///
    /// # Errors
    /// Fails when the shape's element count differs from the value count.
    pub fn from_i16(values: &[i16], shape: &[i32]) -> Result<Self, MlxCError> {
        Self::from_host_values(values, shape, MlxDtype::Int16, "create an MLX int16 array")
    }

    /// Creates an int64 array from host values.
    ///
    /// # Errors
    /// Fails when the shape's element count differs from the value count.
    pub fn from_i64(values: &[i64], shape: &[i32]) -> Result<Self, MlxCError> {
        Self::from_host_values(values, shape, MlxDtype::Int64, "create an MLX int64 array")
    }

    /// Creates a float64 array from host values. The GPU rejects float64, so
    /// operate on it through a context from `MlxBindingsContext::on_cpu`.
    ///
    /// # Errors
    /// Fails when the shape's element count differs from the value count.
    pub fn from_f64(values: &[f64], shape: &[i32]) -> Result<Self, MlxCError> {
        Self::from_host_values(
            values,
            shape,
            MlxDtype::Float64,
            "create an MLX float64 array",
        )
    }

    /// Creates a boolean array from host values.
    ///
    /// # Errors
    /// Fails when the shape's element count differs from the value count.
    pub fn from_bools(values: &[bool], shape: &[i32]) -> Result<Self, MlxCError> {
        Self::from_host_values(values, shape, MlxDtype::Bool, "create an MLX boolean array")
    }

    /// Creates a float16 array from raw IEEE binary16 bits, the same form
    /// `to_vec_f16_bits` reads back.
    ///
    /// # Errors
    /// Fails when the shape's element count differs from the value count.
    pub fn from_f16_bits(bits: &[u16], shape: &[i32]) -> Result<Self, MlxCError> {
        Self::from_host_values(
            bits,
            shape,
            MlxDtype::Float16,
            "create an MLX float16 array",
        )
    }

    /// Creates a bfloat16 array from raw bits, the same form `to_vec_bf16_bits`
    /// reads back.
    ///
    /// # Errors
    /// Fails when the shape's element count differs from the value count.
    pub fn from_bf16_bits(bits: &[u16], shape: &[i32]) -> Result<Self, MlxCError> {
        Self::from_host_values(
            bits,
            shape,
            MlxDtype::BFloat16,
            "create an MLX bfloat16 array",
        )
    }

    /// Creates a complex array from `(real, imaginary)` pairs, the same form
    /// `to_vec_complex` reads back.
    ///
    /// # Errors
    /// Fails when the shape's element count differs from the value count.
    pub fn from_complex(values: &[(f32, f32)], shape: &[i32]) -> Result<Self, MlxCError> {
        // Rust leaves a tuple's field layout unspecified, so each pair moves
        // into MLX-C's own layout before the bytes are handed over.
        let mut complex_values = Vec::with_capacity(values.len());
        for (real_part, imaginary_part) in values {
            complex_values.push(raw::mlx_complex64_t {
                re: *real_part,
                im: *imaginary_part,
            });
        }
        Self::from_host_values(
            &complex_values,
            shape,
            MlxDtype::Complex64,
            "create an MLX complex array",
        )
    }
}
