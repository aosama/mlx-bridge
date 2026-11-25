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

/// MLX's coarse classification of an element type, as MLX's `kindof`
/// reports it. `BFloat16` has a kind of its own because its bit layout is not
/// an IEEE floating-point format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MlxDtypeKind {
    /// `Bool`.
    Bool,
    /// The `UInt*` types.
    UnsignedInteger,
    /// The `Int*` types.
    SignedInteger,
    /// `Float16`, `Float32`, and `Float64`.
    Floating,
    /// `Complex64`.
    Complex,
    /// `BFloat16`.
    BrainFloating,
}

impl MlxDtypeKind {
    fn from_raw(raw_kind: raw::mlxrs_dtype_kind) -> Result<Self, MlxCError> {
        Ok(match raw_kind {
            raw::mlxrs_dtype_kind__MLXRS_DTYPE_KIND_BOOL => Self::Bool,
            raw::mlxrs_dtype_kind__MLXRS_DTYPE_KIND_UNSIGNED_INTEGER => Self::UnsignedInteger,
            raw::mlxrs_dtype_kind__MLXRS_DTYPE_KIND_SIGNED_INTEGER => Self::SignedInteger,
            raw::mlxrs_dtype_kind__MLXRS_DTYPE_KIND_FLOATING => Self::Floating,
            raw::mlxrs_dtype_kind__MLXRS_DTYPE_KIND_COMPLEX => Self::Complex,
            raw::mlxrs_dtype_kind__MLXRS_DTYPE_KIND_BRAIN_FLOATING => Self::BrainFloating,
            unknown_kind => {
                return Err(MlxCError {
                    operation: "read an MLX element type kind",
                    description: format!(
                        "MLX returned an unknown element type kind {unknown_kind}"
                    ),
                });
            }
        })
    }
}

/// A node in MLX's element type hierarchy, for [`MlxDtype::is_in_category`].
/// `Generic` contains every type, `Number` every type but `Bool`, `Inexact`
/// the floating and complex types, and `Integer` both integer families.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MlxDtypeCategory {
    /// `Complex64`.
    ComplexFloating,
    /// The real floating-point types, `BFloat16` included.
    Floating,
    /// `Floating` and `ComplexFloating`.
    Inexact,
    /// The `Int*` types.
    SignedInteger,
    /// The `UInt*` types.
    UnsignedInteger,
    /// `SignedInteger` and `UnsignedInteger`.
    Integer,
    /// `Inexact` and `Integer`.
    Number,
    /// Every element type.
    Generic,
}

impl MlxDtypeCategory {
    const fn to_raw(self) -> raw::mlxrs_dtype_category {
        match self {
            Self::ComplexFloating => {
                raw::mlxrs_dtype_category__MLXRS_DTYPE_CATEGORY_COMPLEX_FLOATING
            }
            Self::Floating => raw::mlxrs_dtype_category__MLXRS_DTYPE_CATEGORY_FLOATING,
            Self::Inexact => raw::mlxrs_dtype_category__MLXRS_DTYPE_CATEGORY_INEXACT,
            Self::SignedInteger => raw::mlxrs_dtype_category__MLXRS_DTYPE_CATEGORY_SIGNED_INTEGER,
            Self::UnsignedInteger => {
                raw::mlxrs_dtype_category__MLXRS_DTYPE_CATEGORY_UNSIGNED_INTEGER
            }
            Self::Integer => raw::mlxrs_dtype_category__MLXRS_DTYPE_CATEGORY_INTEGER,
            Self::Number => raw::mlxrs_dtype_category__MLXRS_DTYPE_CATEGORY_NUMBER,
            Self::Generic => raw::mlxrs_dtype_category__MLXRS_DTYPE_CATEGORY_GENERIC,
        }
    }
}

impl MlxDtype {
    /// The element type this type and `other` promote to together, the
    /// type MLX gives the result of a binary operation on the two.
    ///
    /// # Errors
    /// Returns the captured MLX description when the query fails.
    pub fn promote_with(self, other: Self) -> Result<Self, MlxCError> {
        let mut promoted_dtype = raw::mlx_dtype__MLX_BOOL;
        // SAFETY: The output pointer is valid writable storage and both
        // element types are plain values.
        let status =
            unsafe { raw::mlxrs_promote_types(&mut promoted_dtype, self.to_raw(), other.to_raw()) };
        check_status(status, "promote two MLX element types")?;
        Ok(Self::from_raw(promoted_dtype))
    }

    /// This element type's kind.
    ///
    /// # Errors
    /// Returns the captured MLX description when the query fails.
    pub fn kind(self) -> Result<MlxDtypeKind, MlxCError> {
        let mut raw_kind = raw::mlxrs_dtype_kind__MLXRS_DTYPE_KIND_BOOL;
        // SAFETY: The output pointer is valid writable storage and the
        // element type is a plain value.
        let status = unsafe { raw::mlxrs_kindof(&mut raw_kind, self.to_raw()) };
        check_status(status, "read an MLX element type kind")?;
        MlxDtypeKind::from_raw(raw_kind)
    }

    /// Whether this element type falls under `category` in MLX's hierarchy.
    ///
    /// # Errors
    /// Returns the captured MLX description when the query fails.
    pub fn is_in_category(self, category: MlxDtypeCategory) -> Result<bool, MlxCError> {
        let mut is_in_category = false;
        // SAFETY: The output pointer is valid writable storage and both
        // arguments are plain values.
        let status =
            unsafe { raw::mlxrs_issubdtype(&mut is_in_category, self.to_raw(), category.to_raw()) };
        check_status(status, "test an MLX element type's category")?;
        Ok(is_in_category)
    }

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
