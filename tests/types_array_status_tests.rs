//! Tier 5 contracts for the array status queries: evaluation progress, storage
//! layout, printing, and the complex readback.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{float_array, gpu_context};
use mlx_c_rust::{MlxArray, MlxDtype};

#[test]
fn should_report_availability_only_once_a_graph_has_been_evaluated() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0], &[2]);
    let doubled = context.add(&values, &values).unwrap();

    assert!(
        !doubled.is_available().unwrap(),
        "the graph has not run yet"
    );

    doubled.evaluate().unwrap();

    assert!(doubled.is_available().unwrap());
}

#[test]
fn should_finish_scheduled_work_when_waiting_on_an_array() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0], &[2]);
    let doubled = context.add(&values, &values).unwrap();

    context.async_eval_arrays(&[&doubled]).unwrap();
    doubled.wait().unwrap();

    assert!(doubled.is_available().unwrap());
}

#[test]
fn should_compute_real_values_when_waiting_on_an_unscheduled_array() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0], &[2]);
    let doubled = context.add(&values, &values).unwrap();

    // MLX-C's own wait would mark this available with no data behind it.
    doubled.wait().unwrap();

    assert!(doubled.is_available().unwrap());
    assert_eq!(doubled.to_vec_f32().unwrap(), vec![2.0, 4.0]);
    let built_on_top = context.add(&doubled, &doubled).unwrap();
    assert_eq!(built_on_top.to_vec_f32().unwrap(), vec![4.0, 8.0]);
}

#[test]
fn should_refuse_status_queries_on_an_empty_handle() {
    gpu_context();
    let empty = MlxArray::empty();

    assert!(empty.is_available().is_err());
    assert!(empty.wait().is_err());
    assert!(empty.is_contiguous().is_err());
    assert!(empty.is_col_contiguous().is_err());
    assert!(empty.to_string_lossy().is_err());
    assert!(empty.to_vec_complex().is_err());
}

#[test]
fn should_distinguish_row_and_column_contiguous_layouts() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let transposed = context.transpose(&values).unwrap();

    assert!(values.is_row_contiguous().unwrap());
    assert!(!values.is_col_contiguous().unwrap());
    assert!(values.is_contiguous().unwrap());
    assert!(!transposed.is_row_contiguous().unwrap());
    assert!(transposed.is_col_contiguous().unwrap());
    assert!(transposed.is_contiguous().unwrap());
}

#[test]
fn should_report_a_strided_slice_as_not_contiguous() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0, 3.0, 4.0], &[4]);
    values.evaluate().unwrap();

    let every_second = context.slice(&values, &[0], &[4], &[2]).unwrap();

    assert!(!every_second.is_contiguous().unwrap());
}

#[test]
fn should_render_an_array_with_its_values() {
    gpu_context();
    let values = float_array(&[1.5, 2.5], &[2]);

    let description = values.to_string_lossy().unwrap();

    assert!(description.contains("1.5"), "{description}");
    assert!(description.contains("2.5"), "{description}");
}

#[test]
fn should_read_a_reversed_complex_view_in_its_logical_order() {
    let context = gpu_context();
    let values = MlxArray::from_complex(&[(1.0, 2.0), (3.0, 4.0)], &[2]).unwrap();
    values.evaluate().unwrap();

    let reversed = context.flip(&values).unwrap();

    assert_eq!(reversed.dtype(), MlxDtype::Complex64);
    assert_eq!(
        reversed.to_vec_complex().unwrap(),
        vec![(3.0, 4.0), (1.0, 2.0)]
    );
}

#[test]
fn should_refuse_a_complex_read_of_real_storage() {
    gpu_context();

    assert!(float_array(&[1.0], &[1]).to_vec_complex().is_err());
    assert!(
        MlxArray::from_complex(&[(1.0, 2.0)], &[1])
            .unwrap()
            .to_vec_f32()
            .is_err()
    );
}
