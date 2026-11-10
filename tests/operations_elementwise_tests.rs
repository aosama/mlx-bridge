//! Tier 4 marshalling evidence for the elementwise families: argument order,
//! optional arguments, and each wrapper reaching its own C function. Broadcasting,
//! promotion, and edge-value semantics belong to MLX's own suite and stay there.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{
    FLOAT_TOLERANCE, assert_floats_close, bools_of, float_array, floats_of, gpu_context, int_array,
    ints_of,
};
use mlx_bridge::MlxArray;

#[test]
fn should_keep_argument_order_for_non_commutative_operations() {
    let context = gpu_context();
    let left = float_array(&[8.0], &[1]);
    let right = float_array(&[2.0], &[1]);

    assert_eq!(
        floats_of(&context.subtract(&left, &right).unwrap()),
        vec![6.0]
    );
    assert_eq!(
        floats_of(&context.divide(&left, &right).unwrap()),
        vec![4.0]
    );
    assert_eq!(
        floats_of(&context.power(&left, &right).unwrap()),
        vec![64.0]
    );
    assert_eq!(bools_of(&context.less(&left, &right).unwrap()), vec![false]);
    assert_eq!(
        bools_of(&context.greater(&left, &right).unwrap()),
        vec![true]
    );
}

#[test]
fn should_bind_each_classifier_to_its_own_c_function() {
    let context = gpu_context();
    let values = float_array(&[1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY], &[4]);

    assert_eq!(
        bools_of(&context.isnan(&values).unwrap()),
        vec![false, true, false, false]
    );
    assert_eq!(
        bools_of(&context.isinf(&values).unwrap()),
        vec![false, false, true, true]
    );
    assert_eq!(
        bools_of(&context.isposinf(&values).unwrap()),
        vec![false, false, true, false]
    );
    assert_eq!(
        bools_of(&context.isneginf(&values).unwrap()),
        vec![false, false, false, true]
    );
    assert_eq!(
        bools_of(&context.isfinite(&values).unwrap()),
        vec![true, false, false, false]
    );
}

#[test]
fn should_replace_non_finite_values_with_the_given_substitutes() {
    let context = gpu_context();
    let values = float_array(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY], &[3]);

    let replaced = context
        .nan_to_num(&values, 1.0, Some(2.0), Some(3.0))
        .unwrap();

    assert_eq!(floats_of(&replaced), vec![1.0, 2.0, 3.0]);
}

#[test]
fn should_select_between_arrays_by_condition() {
    let context = gpu_context();
    let left = float_array(&[1.0, 2.0], &[2]);
    let right = float_array(&[3.0, 1.0], &[2]);
    let condition = context.greater(&left, &right).unwrap();

    let selected = context.where_select(&condition, &left, &right).unwrap();

    assert_eq!(floats_of(&selected), vec![3.0, 2.0]);
}

#[test]
fn should_evaluate_logical_truth_tables_in_argument_order() {
    let context = gpu_context();
    let left = context
        .greater(
            &float_array(&[1.0, 1.0, 0.0, 0.0], &[4]),
            &MlxArray::from_scalar_f32(0.5),
        )
        .unwrap();
    let right = context
        .greater(
            &float_array(&[1.0, 0.0, 1.0, 0.0], &[4]),
            &MlxArray::from_scalar_f32(0.5),
        )
        .unwrap();

    assert_eq!(
        bools_of(&context.logical_and(&left, &right).unwrap()),
        vec![true, false, false, false]
    );
    assert_eq!(
        bools_of(&context.logical_or(&left, &right).unwrap()),
        vec![true, true, true, false]
    );
    assert_eq!(
        bools_of(&context.logical_not(&left).unwrap()),
        vec![false, false, true, true]
    );
}

#[test]
fn should_apply_bitwise_operations_to_integers_in_argument_order() {
    let context = gpu_context();
    let left = int_array(&[0b1100], &[1]);
    let right = int_array(&[0b1010], &[1]);
    let shift = int_array(&[1], &[1]);

    assert_eq!(
        ints_of(&context.bitwise_and(&left, &right).unwrap()),
        vec![0b1000]
    );
    assert_eq!(
        ints_of(&context.bitwise_or(&left, &right).unwrap()),
        vec![0b1110]
    );
    assert_eq!(
        ints_of(&context.bitwise_xor(&left, &right).unwrap()),
        vec![0b0110]
    );
    assert_eq!(
        ints_of(&context.left_shift(&left, &shift).unwrap()),
        vec![0b11000]
    );
    assert_eq!(
        ints_of(&context.right_shift(&left, &shift).unwrap()),
        vec![0b0110]
    );
}

#[test]
fn should_compare_arrays_through_the_tolerance_arguments() {
    let context = gpu_context();
    let left = float_array(&[1.0, f32::NAN], &[2]);
    let right = float_array(&[1.001, f32::NAN], &[2]);

    let loose = context.isclose(&left, &right, 0.0, 0.01, true).unwrap();
    let tight = context.isclose(&left, &right, 0.0, 1e-6, false).unwrap();

    assert_eq!(bools_of(&loose), vec![true, true]);
    assert_eq!(bools_of(&tight), vec![false, false]);
}

#[test]
fn should_multiply_matrices_in_argument_order() {
    let context = gpu_context();
    let matrix = float_array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let column = float_array(&[1.0, 1.0], &[2, 1]);

    let product = context.matmul(&matrix, &column).unwrap();

    assert_eq!(product.shape(), vec![2, 1]);
    assert_eq!(floats_of(&product), vec![3.0, 7.0]);
    assert!(context.matmul(&column, &column).is_err());
}

#[test]
fn should_combine_vectors_through_inner_and_outer_products() {
    let context = gpu_context();
    let left = float_array(&[1.0, 2.0], &[2]);
    let right = float_array(&[3.0, 4.0], &[2]);

    assert_eq!(
        floats_of(&context.inner(&left, &right).unwrap()),
        vec![11.0]
    );
    assert_eq!(
        floats_of(&context.outer(&left, &right).unwrap()),
        vec![3.0, 4.0, 6.0, 8.0]
    );
}

#[test]
fn should_keep_argument_order_for_integer_division_and_remainder() {
    let context = gpu_context();
    let dividend = int_array(&[7], &[1]);
    let divisor = int_array(&[2], &[1]);

    assert_eq!(
        ints_of(&context.floor_divide(&dividend, &divisor).unwrap()),
        vec![3]
    );
    assert_eq!(
        ints_of(&context.remainder(&dividend, &divisor).unwrap()),
        vec![1]
    );
}

#[test]
fn should_forward_the_decimals_argument_when_rounding() {
    let context = gpu_context();

    let hundredths = context.round(&float_array(&[1.234], &[1]), 2).unwrap();

    assert_floats_close(&floats_of(&hundredths), &[1.23], FLOAT_TOLERANCE);
}

#[test]
fn should_map_absent_substitutes_to_the_mlx_defaults_rather_than_zero() {
    let context = gpu_context();
    let values = float_array(&[f32::INFINITY], &[1]);

    let replaced = floats_of(&context.nan_to_num(&values, 0.0, None, None).unwrap());

    assert!(
        replaced[0] > 1.0,
        "None must not arrive as a present zero: {replaced:?}"
    );
}
