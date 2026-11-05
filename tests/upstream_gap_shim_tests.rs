//! Gap-shim contracts: the repository's own mlxrs_* C ABI over the MLX
//! functions the vendored MLX-C pin has not wrapped, exercised end to end
//! through the bridge — the shim compiles in the native prefix, binds
//! through the combined parse, and behaves like any MLX-C family: real
//! values, captured errors, owned handles. These run serialized with the
//! rest of the native-linked tree (`cargo native-test`).

use mlx_c_rust::{MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

#[test]
fn should_compute_minmax_over_axes_with_both_outputs() {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new().expect("the bindings context should initialize");

    let values = context
        .array_from_f32(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3])
        .expect("the input array should be valid");

    let (minima, maxima) = context
        .minmax_axes(&values, &[1], false)
        .expect("the gap shim minmax over axes should build on the GPU stream");
    let minima_values = minima
        .to_vec_f32()
        .expect("the evaluated minima should copy back to host memory");
    let maxima_values = maxima
        .to_vec_f32()
        .expect("the evaluated maxima should copy back to host memory");

    assert_eq!(minima_values, vec![1.0, 1.0]);
    assert_eq!(maxima_values, vec![4.0, 9.0]);
}

#[test]
fn should_keep_minmax_axis_dimensions_when_requested() {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new().expect("the bindings context should initialize");

    let values = context
        .array_from_f32(&[2.0, 8.0, 4.0, 6.0], &[2, 2])
        .expect("the input array should be valid");

    let (minima, maxima) = context
        .minmax_axis(&values, 1, true)
        .expect("the gap shim minmax along one axis should build on the GPU stream");

    assert_eq!(
        minima.shape(),
        vec![2, 1],
        "keepdims must preserve the reduced axis as size one"
    );
    assert_eq!(maxima.shape(), vec![2, 1]);
}

#[test]
fn should_surface_the_captured_error_for_an_out_of_range_axis() {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new().expect("the bindings context should initialize");

    let values = context
        .array_from_f32(&[1.0, 2.0], &[2])
        .expect("the input array should be valid");

    let captured_error = context
        .minmax_axis(&values, 7, false)
        .expect_err("an out-of-range axis must fail through the captured-error channel");

    let MlxCError {
        operation,
        description,
    } = captured_error;
    assert!(
        operation.contains("minmax"),
        "the captured error must name the shim operation, got: {operation}"
    );
    assert!(
        !description.is_empty(),
        "MLX's reported description must ride along with the shim failure"
    );
}
