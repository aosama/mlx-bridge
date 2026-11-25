//! Contracts for the element type queries: each field of MLX's limits lands in
//! the field of the same name, and promotion answers come back as `MlxDtype`.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{float_array, gpu_context, int_array, vector_of};
use mlx_bridge::{MlxArray, MlxDtype, MlxDtypeCategory, MlxDtypeKind, result_type};

#[test]
fn should_place_each_float_limit_in_its_own_field() {
    gpu_context();

    let limits = MlxDtype::Float32.float_info().unwrap();

    assert_eq!(limits.bits, 32);
    assert_eq!(limits.max, f64::from(f32::MAX));
    assert_eq!(limits.min, f64::from(f32::MIN));
    assert_eq!(limits.eps, f64::from(f32::EPSILON));
    assert_eq!(limits.smallest_normal, f64::from(f32::MIN_POSITIVE));
}

#[test]
fn should_read_the_limits_of_each_requested_float_type() {
    gpu_context();

    assert_eq!(MlxDtype::Float16.float_info().unwrap().bits, 16);
    assert_eq!(MlxDtype::Float64.float_info().unwrap().bits, 64);
    assert_eq!(MlxDtype::BFloat16.float_info().unwrap().bits, 16);
}

#[test]
fn should_refuse_float_limits_for_an_integer_type() {
    gpu_context();

    assert!(MlxDtype::Int32.float_info().is_err());
    assert!(MlxDtype::Bool.float_info().is_err());
}

#[test]
fn should_map_the_promoted_type_back_to_the_rust_enum() {
    gpu_context();
    let integers = int_array(&[1], &[1]);
    let floats = float_array(&[1.0], &[1]);
    let bytes = MlxArray::from_u8(&[1], &[1]).unwrap();

    assert_eq!(
        result_type(&vector_of(&[&integers, &floats])).unwrap(),
        MlxDtype::Float32
    );
    assert_eq!(result_type(&vector_of(&[&bytes])).unwrap(), MlxDtype::UInt8);
}

#[test]
fn should_start_promotion_from_boolean_for_an_empty_vector() {
    gpu_context();

    assert_eq!(result_type(&vector_of(&[])).unwrap(), MlxDtype::Bool);
}

#[test]
fn should_promote_two_element_types_to_the_type_that_holds_both() {
    gpu_context();

    // Each pair promotes to a type neither argument is, which only a call
    // that received both arguments in order can return.
    assert_eq!(
        MlxDtype::Int8.promote_with(MlxDtype::UInt8).unwrap(),
        MlxDtype::Int16
    );
    assert_eq!(
        MlxDtype::Float16.promote_with(MlxDtype::BFloat16).unwrap(),
        MlxDtype::Float32
    );
}

#[test]
fn should_promote_a_type_with_itself_to_itself() {
    // Boundary: promotion's identity, including Bool at the bottom of the
    // lattice and Complex64 at the top.
    gpu_context();

    for dtype in [MlxDtype::Bool, MlxDtype::BFloat16, MlxDtype::Complex64] {
        assert_eq!(dtype.promote_with(dtype).unwrap(), dtype);
    }
}

#[test]
fn should_report_each_element_type_kind() {
    gpu_context();

    assert_eq!(MlxDtype::Bool.kind().unwrap(), MlxDtypeKind::Bool);
    assert_eq!(
        MlxDtype::UInt32.kind().unwrap(),
        MlxDtypeKind::UnsignedInteger
    );
    assert_eq!(MlxDtype::Int64.kind().unwrap(), MlxDtypeKind::SignedInteger);
    assert_eq!(MlxDtype::Float16.kind().unwrap(), MlxDtypeKind::Floating);
    assert_eq!(MlxDtype::Complex64.kind().unwrap(), MlxDtypeKind::Complex);
    // BFloat16 is the one floating type MLX does not classify as Floating.
    assert_eq!(
        MlxDtype::BFloat16.kind().unwrap(),
        MlxDtypeKind::BrainFloating
    );
}

#[test]
fn should_place_element_types_in_their_categories() {
    gpu_context();

    assert!(
        MlxDtype::BFloat16
            .is_in_category(MlxDtypeCategory::Floating)
            .unwrap()
    );
    assert!(
        MlxDtype::Complex64
            .is_in_category(MlxDtypeCategory::Inexact)
            .unwrap()
    );
    assert!(
        MlxDtype::UInt8
            .is_in_category(MlxDtypeCategory::Integer)
            .unwrap()
    );
    assert!(
        !MlxDtype::Int32
            .is_in_category(MlxDtypeCategory::Inexact)
            .unwrap()
    );
    assert!(
        !MlxDtype::UInt8
            .is_in_category(MlxDtypeCategory::SignedInteger)
            .unwrap()
    );
}

#[test]
fn should_keep_bool_out_of_every_category_but_generic() {
    // Boundary: Bool is the one type outside Number, and Generic holds every
    // type there is.
    gpu_context();

    assert!(
        !MlxDtype::Bool
            .is_in_category(MlxDtypeCategory::Number)
            .unwrap()
    );
    for dtype in [
        MlxDtype::Bool,
        MlxDtype::Int16,
        MlxDtype::Float64,
        MlxDtype::Complex64,
    ] {
        assert!(dtype.is_in_category(MlxDtypeCategory::Generic).unwrap());
    }
}
