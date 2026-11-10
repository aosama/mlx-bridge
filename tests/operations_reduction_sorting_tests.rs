//! Tier 4 marshalling evidence for reductions, cumulative operations, and
//! sorting: axis and keepdims arguments, and the reverse and inclusive flags
//! reaching the right C parameters.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{
    FLOAT_TOLERANCE, assert_floats_close, bools_of, float_array, floats_of, gpu_context,
};
use mlx_bridge::MlxArray;

fn two_by_three() -> MlxArray {
    float_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
}

#[test]
fn should_reduce_over_every_axis_by_default() {
    let context = gpu_context();
    let values = two_by_three();

    let total = context.sum(&values, false).unwrap();

    assert_eq!(total.shape(), Vec::<i32>::new());
    assert_eq!(floats_of(&total), vec![21.0]);
}

#[test]
fn should_keep_reduced_dimensions_when_requested() {
    let context = gpu_context();
    let values = two_by_three();

    assert_eq!(context.sum(&values, true).unwrap().shape(), vec![1, 1]);
    assert_eq!(
        context.sum_axis(&values, 0, true).unwrap().shape(),
        vec![1, 3]
    );
    assert_eq!(
        context.sum_axis(&values, 0, false).unwrap().shape(),
        vec![3]
    );
}

#[test]
fn should_reduce_along_the_requested_axes() {
    let context = gpu_context();
    let values = two_by_three();

    assert_eq!(
        floats_of(&context.sum_axis(&values, 1, false).unwrap()),
        vec![6.0, 15.0]
    );
    assert_eq!(
        floats_of(&context.sum_axis(&values, -2, false).unwrap()),
        vec![5.0, 7.0, 9.0]
    );
    assert_eq!(
        floats_of(&context.max_axes(&values, &[0, 1], false).unwrap()),
        vec![6.0]
    );
    assert_eq!(
        floats_of(&context.min_axis(&values, 0, false).unwrap()),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        floats_of(&context.mean_axes(&values, &[1], false).unwrap()),
        vec![2.0, 5.0]
    );
}

#[test]
fn should_apply_the_degrees_of_freedom_argument_to_variance_and_deviation() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0, 3.0, 4.0], &[4]);

    assert_floats_close(
        &floats_of(&context.var(&values, false, 0).unwrap()),
        &[1.25],
        FLOAT_TOLERANCE,
    );
    assert_floats_close(
        &floats_of(&context.var(&values, false, 1).unwrap()),
        &[5.0 / 3.0],
        FLOAT_TOLERANCE,
    );
    assert_floats_close(
        &floats_of(&context.std(&values, false, 0).unwrap()),
        &[1.25_f32.sqrt()],
        FLOAT_TOLERANCE,
    );
}

#[test]
fn should_locate_extremes_along_an_axis() {
    let context = gpu_context();
    let values = float_array(&[1.0, 9.0, 8.0, 2.0], &[2, 2]);

    let largest = context.argmax_axis(&values, 1).unwrap();
    let smallest = context.argmin_axis(&values, 1, true).unwrap();

    assert_eq!(largest.to_vec_u32().unwrap(), vec![1, 0]);
    assert_eq!(smallest.shape(), vec![2, 1]);
    assert_eq!(smallest.to_vec_u32().unwrap(), vec![0, 1]);
}

#[test]
fn should_reduce_truth_values_across_and_along_axes() {
    let context = gpu_context();
    let truth_values = context
        .greater(
            &float_array(&[1.0, 0.0, 1.0, 1.0], &[2, 2]),
            &MlxArray::from_scalar_f32(0.5),
        )
        .unwrap();

    assert_eq!(
        bools_of(&context.all(&truth_values, false).unwrap()),
        vec![false]
    );
    assert_eq!(
        bools_of(&context.any(&truth_values, false).unwrap()),
        vec![true]
    );
    assert_eq!(
        bools_of(&context.all_axis(&truth_values, 1, false).unwrap()),
        vec![false, true]
    );
}

#[test]
fn should_honor_the_reverse_and_inclusive_flags_of_cumulative_operations() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0, 3.0], &[3]);

    assert_eq!(
        floats_of(&context.cumsum(&values, 0, false, true).unwrap()),
        vec![1.0, 3.0, 6.0]
    );
    assert_eq!(
        floats_of(&context.cumsum(&values, 0, true, true).unwrap()),
        vec![6.0, 5.0, 3.0]
    );
    assert_eq!(
        floats_of(&context.cumsum(&values, 0, false, false).unwrap()),
        vec![0.0, 1.0, 3.0]
    );
    assert_eq!(
        floats_of(&context.cummax_axis(&values, 0, false, true).unwrap()),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        floats_of(&context.cumprod_axis(&values, 0, false, true, None).unwrap()),
        vec![1.0, 2.0, 6.0]
    );
}

#[test]
fn should_sort_and_report_the_sorting_permutation() {
    let context = gpu_context();
    let values = float_array(&[3.0, 1.0, 2.0], &[3]);

    assert_eq!(
        floats_of(&context.sort(&values).unwrap()),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        context.argsort(&values).unwrap().to_vec_u32().unwrap(),
        vec![1, 2, 0]
    );
}

#[test]
fn should_sort_along_the_requested_axis() {
    let context = gpu_context();
    let values = float_array(&[4.0, 3.0, 2.0, 1.0], &[2, 2]);

    let along_rows = context.sort_axis(&values, 1).unwrap();
    let along_columns = context.sort_axis(&values, 0).unwrap();

    assert_eq!(floats_of(&along_rows), vec![3.0, 4.0, 1.0, 2.0]);
    assert_eq!(floats_of(&along_columns), vec![2.0, 1.0, 4.0, 3.0]);
}

#[test]
fn should_return_the_largest_values_through_top_k() {
    let context = gpu_context();
    let values = float_array(&[1.0, 5.0, 3.0, 4.0], &[4]);

    let mut largest = floats_of(&context.topk(&values, 2).unwrap());
    largest.sort_by(f32::total_cmp);

    assert_eq!(largest, vec![4.0, 5.0]);
}

#[test]
fn should_place_the_kth_element_at_its_sorted_position_through_partition() {
    let context = gpu_context();
    let values = float_array(&[4.0, 1.0, 3.0, 2.0], &[4]);

    let partitioned = floats_of(&context.partition(&values, 1).unwrap());

    assert_eq!(partitioned[1], 2.0);
    assert!(partitioned[0] <= 2.0);
}

#[test]
fn should_bind_each_extreme_locator_to_its_own_c_function() {
    let context = gpu_context();
    let values = float_array(&[2.0, 3.0, 1.0], &[3]);

    assert_eq!(
        context.argmax(&values, false).unwrap().item_u32().unwrap(),
        1
    );
    assert_eq!(
        context.argmin(&values, false).unwrap().item_u32().unwrap(),
        2
    );
}
