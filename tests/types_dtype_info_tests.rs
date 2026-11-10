//! Contracts for the element type queries: each field of MLX's limits lands in
//! the field of the same name, and promotion answers come back as `MlxDtype`.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{float_array, gpu_context, int_array, vector_of};
use mlx_bridge::{MlxArray, MlxDtype, result_type};

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
