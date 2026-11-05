//! Tier 4 marshalling evidence for the creation and shape families: argument
//! order, default axes, and result ownership. MLX owns the numerics; these
//! values only prove the arguments arrived where they were sent.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{
    FLOAT_TOLERANCE, array_at, assert_floats_close, float_array, floats_of, gpu_context, int_array,
    ints_of, vector_of,
};
use mlx_c_rust::{MlxArray, MlxDtype};

#[test]
fn should_create_ranges_with_the_requested_bounds_and_stride() {
    let context = gpu_context();

    let integers = context.arange_i32(2, 5).unwrap();
    let floats = context.arange_f32(0.0, 1.0, 0.25).unwrap();

    assert_eq!(ints_of(&integers), vec![2, 3, 4]);
    assert_eq!(floats_of(&floats), vec![0.0, 0.25, 0.5, 0.75]);
}

#[test]
fn should_fill_arrays_with_the_requested_shape_value_and_element_type() {
    let context = gpu_context();

    let zeros = context.zeros(&[2], MlxDtype::Int32).unwrap();
    let ones = context.ones(&[2], MlxDtype::Float32).unwrap();
    let filled = context.full(&[3], 2.5, MlxDtype::Float32).unwrap();

    assert_eq!(ints_of(&zeros), vec![0, 0]);
    assert_eq!(floats_of(&ones), vec![1.0, 1.0]);
    assert_eq!(floats_of(&filled), vec![2.5, 2.5, 2.5]);
}

#[test]
fn should_derive_like_arrays_from_their_template() {
    let context = gpu_context();
    let template = float_array(&[5.0, 6.0], &[2]);

    assert_eq!(
        floats_of(&context.zeros_like(&template).unwrap()),
        vec![0.0, 0.0]
    );
    assert_eq!(
        floats_of(&context.ones_like(&template).unwrap()),
        vec![1.0, 1.0]
    );
    assert_eq!(
        context
            .ones_like_dtype(&template, MlxDtype::Int32)
            .unwrap()
            .dtype(),
        MlxDtype::Int32
    );
}

#[test]
fn should_place_the_diagonal_offset_in_eye() {
    let context = gpu_context();

    let shifted_eye = context.eye(2, 3, 1, MlxDtype::Int32).unwrap();

    assert_eq!(shifted_eye.shape(), vec![2, 3]);
    assert_eq!(ints_of(&shifted_eye), vec![0, 1, 0, 0, 0, 1]);
}

#[test]
fn should_include_both_endpoints_in_linspace() {
    let context = gpu_context();

    let spaced_values = context.linspace(0.0, 1.0, 3, MlxDtype::Float32).unwrap();

    assert_floats_close(
        &floats_of(&spaced_values),
        &[0.0, 0.5, 1.0],
        FLOAT_TOLERANCE,
    );
}

#[test]
fn should_infer_a_reshape_dimension_and_preserve_row_order() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);

    let reshaped = context.reshape(&values, &[2, -1]).unwrap();

    assert_eq!(reshaped.shape(), vec![2, 3]);
    assert_eq!(floats_of(&reshaped), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn should_apply_transpose_axes_in_the_given_order() {
    let context = gpu_context();
    let values = context.zeros(&[2, 3, 4], MlxDtype::Float32).unwrap();

    let permuted = context.transpose_axes(&values, &[2, 0, 1]).unwrap();

    assert_eq!(permuted.shape(), vec![4, 2, 3]);
}

#[test]
fn should_add_and_remove_axes_at_the_requested_positions() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0], &[2]);

    let expanded = context.expand_dims(&values, 0).unwrap();
    let expanded_from_end = context.expand_dims(&values, -1).unwrap();
    let squeezed = context.squeeze_axis(&expanded, 0).unwrap();

    assert_eq!(expanded.shape(), vec![1, 2]);
    assert_eq!(expanded_from_end.shape(), vec![2, 1]);
    assert_eq!(squeezed.shape(), vec![2]);
}

#[test]
fn should_concatenate_and_stack_along_the_requested_axis() {
    let context = gpu_context();
    let first = float_array(&[1.0, 2.0], &[2]);
    let second = float_array(&[3.0, 4.0], &[2]);

    let concatenated = context.concatenate_axis(&[&first, &second], 0).unwrap();
    let stacked = context.stack_axis(&[&first, &second], 1).unwrap();
    let from_vector = context.concatenate(&vector_of(&[&first, &second])).unwrap();

    assert_eq!(floats_of(&concatenated), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(stacked.shape(), vec![2, 2]);
    assert_eq!(floats_of(&stacked), vec![1.0, 3.0, 2.0, 4.0]);
    assert_eq!(floats_of(&from_vector), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn should_split_into_equal_parts_and_at_section_indices() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0, 3.0, 4.0], &[4]);

    let halves = context.split(&values, 2, 0).unwrap();
    let sections = context.split_sections(&values, &[1, 3], 0).unwrap();

    assert_eq!(halves.len(), 2);
    assert_eq!(floats_of(&array_at(&halves, 1)), vec![3.0, 4.0]);
    assert_eq!(sections.len(), 3);
    assert_eq!(floats_of(&array_at(&sections, 1)), vec![2.0, 3.0]);
}

#[test]
fn should_broadcast_to_a_larger_shape() {
    let context = gpu_context();

    let broadcast = context
        .broadcast_to(&float_array(&[1.0, 2.0], &[2]), &[2, 2])
        .unwrap();

    assert_eq!(broadcast.shape(), vec![2, 2]);
    assert_eq!(floats_of(&broadcast), vec![1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn should_pad_the_low_and_high_sides_with_the_requested_value() {
    let context = gpu_context();

    let padded = context
        .pad(&float_array(&[1.0], &[1]), &[0], &[1], &[2], 9.0)
        .unwrap();

    assert_eq!(floats_of(&padded), vec![9.0, 1.0, 9.0, 9.0]);
}

#[test]
fn should_update_only_the_requested_slice() {
    let context = gpu_context();
    let source = float_array(&[1.0, 2.0, 3.0], &[3]);
    let update = float_array(&[8.0, 9.0], &[2]);

    let updated = context
        .slice_update(&source, &update, &[1], &[3], &[1])
        .unwrap();

    assert_eq!(floats_of(&updated), vec![1.0, 8.0, 9.0]);
    assert_eq!(floats_of(&source), vec![1.0, 2.0, 3.0]);
}

#[test]
fn should_reject_slice_bounds_of_the_wrong_rank() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0], &[2]);

    assert!(context.slice(&values, &[0, 0], &[1, 1], &[1, 1]).is_err());
}

#[test]
fn should_tile_and_repeat_along_the_requested_axes() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0], &[2]);

    assert_eq!(
        floats_of(&context.tile(&values, &[2]).unwrap()),
        vec![1.0, 2.0, 1.0, 2.0]
    );
    assert_eq!(
        floats_of(&context.repeat_axis(&values, 2, 0).unwrap()),
        vec![1.0, 1.0, 2.0, 2.0]
    );
}

#[test]
fn should_flatten_and_move_the_requested_axes() {
    let context = gpu_context();
    let values = context.zeros(&[2, 3, 4], MlxDtype::Float32).unwrap();

    assert_eq!(context.flatten(&values, 0, 1).unwrap().shape(), vec![6, 4]);
    assert_eq!(
        context.moveaxis(&values, 0, 2).unwrap().shape(),
        vec![3, 4, 2]
    );
    assert_eq!(
        context.swapaxes(&values, 0, 2).unwrap().shape(),
        vec![4, 3, 2]
    );
}

#[test]
fn should_take_values_along_an_axis_at_the_given_indices() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let row_indices = int_array(&[1], &[1]);
    let per_row_indices = int_array(&[1, 0], &[2, 1]);

    let selected_row = context.take_axis(&values, &row_indices, 0).unwrap();
    let per_row = context
        .take_along_axis(&values, &per_row_indices, 1)
        .unwrap();

    assert_eq!(floats_of(&selected_row), vec![3.0, 4.0]);
    assert_eq!(floats_of(&per_row), vec![2.0, 3.0]);
}

#[test]
fn should_put_values_along_an_axis_at_the_given_indices() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0, 3.0], &[3]);
    let indices = int_array(&[2], &[1]);
    let updates = float_array(&[9.0], &[1]);

    let updated = context
        .put_along_axis(&values, &indices, &updates, 0)
        .unwrap();

    assert_eq!(floats_of(&updated), vec![1.0, 2.0, 9.0]);
}

#[test]
fn should_contract_operands_through_einsum_subscripts() {
    let context = gpu_context();
    let matrix = float_array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let trace = context.einsum("ii->", &vector_of(&[&matrix])).unwrap();

    assert_eq!(floats_of(&trace), vec![5.0]);
}

#[test]
fn should_raise_a_scalar_to_the_minimum_rank_through_the_atleast_family() {
    let context = gpu_context();
    let scalar = MlxArray::from_scalar_i32(4);

    assert_eq!(context.atleast_1d(&scalar).unwrap().shape(), vec![1]);
    assert_eq!(context.atleast_2d(&scalar).unwrap().shape(), vec![1, 1]);
    assert_eq!(context.atleast_3d(&scalar).unwrap().shape(), vec![1, 1, 1]);
}
