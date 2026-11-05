//! Tier 3 contracts for error translation: one canonical failure per
//! mechanism rather than per function, each arriving as an `MlxCError` naming
//! the attempted operation and carrying MLX's description, never an abort.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{assert_translated_failure, float_array, floats_of, gpu_context, int_array};
use mlx_c_rust::error::set_closure_error;
use mlx_c_rust::{
    MlxArray, MlxCError, MlxDtype, check_status, clear_captured_mlx_error,
    install_non_terminating_error_handler, take_captured_mlx_error,
};

#[test]
fn should_translate_a_shape_mismatch() {
    let context = gpu_context();
    let left = float_array(&[1.0, 2.0], &[2]);
    let right = float_array(&[1.0, 2.0, 3.0], &[3]);

    let failure = context
        .add(&left, &right)
        .expect_err("shapes (2) and (3) differ");

    assert_translated_failure(&failure, "add MLX arrays");
    assert!(failure.description.contains("broadcast"), "{failure}");
}

#[test]
fn should_translate_an_incompatible_element_type() {
    let context = gpu_context();
    let left = float_array(&[1.0], &[1]);
    let right = float_array(&[1.0], &[1]);

    let failure = context
        .bitwise_and(&left, &right)
        .expect_err("bitwise operations do not accept floats");

    assert_translated_failure(&failure, "");
    assert!(failure.description.contains("bitwise"), "{failure}");
}

#[test]
fn should_translate_an_out_of_bounds_axis() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0], &[2]);

    let failure = context
        .sum_axis(&values, 5, false)
        .expect_err("a one-dimensional array has no axis five");

    assert_translated_failure(&failure, "");
    assert!(failure.description.contains("axis"), "{failure}");
}

#[test]
fn should_translate_an_empty_input_handle() {
    let context = gpu_context();

    let failure = context
        .abs(&MlxArray::empty())
        .expect_err("an empty handle is not an array");

    assert_translated_failure(&failure, "abs");
    assert!(failure.description.contains("non-empty"), "{failure}");
}

#[test]
fn should_translate_an_operation_the_stream_device_does_not_support() {
    let context = gpu_context();
    let matrix = float_array(&[2.0, 0.0, 0.0, 2.0], &[2, 2]);

    let failure = context
        .linalg_inv(&matrix)
        .expect_err("MLX runs matrix inversion on the CPU only");

    assert_translated_failure(&failure, "");
    assert!(failure.description.contains("GPU"), "{failure}");
}

#[test]
fn should_translate_an_element_type_the_stream_device_does_not_support() {
    let context = gpu_context();
    let integers = int_array(&[1, 2], &[2]);

    let failure = context
        .astype(&integers, MlxDtype::Float64)
        .expect_err("the GPU rejects float64 when the graph is built");

    assert_translated_failure(&failure, "cast");
    assert!(failure.description.contains("float64"), "{failure}");
}

#[test]
fn should_succeed_on_the_operation_after_a_failure() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0], &[2]);

    context
        .reshape(&values, &[3])
        .expect_err("the reshape should fail");

    let doubled = context
        .add(&values, &values)
        .expect("the next operation should work");
    assert_eq!(floats_of(&doubled), vec![2.0, 4.0]);
}

#[test]
fn should_not_attribute_one_failure_description_to_the_next_failure() {
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0], &[2]);

    let reshape_failure = context.reshape(&values, &[3]).expect_err("should fail");
    let sum_failure = context
        .sum_axis(&values, 5, false)
        .expect_err("should fail");

    assert!(reshape_failure.description.contains("reshape"));
    assert!(!sum_failure.description.contains("reshape"));
}

#[test]
fn should_accept_a_zero_status() {
    install_non_terminating_error_handler();

    assert!(check_status(0, "a successful call").is_ok());
}

#[test]
fn should_fall_back_to_the_status_when_mlx_reported_no_description() {
    install_non_terminating_error_handler();
    clear_captured_mlx_error();

    let failure = check_status(7, "a silent failure").expect_err("nonzero is a failure");

    assert_eq!(failure.operation, "a silent failure");
    assert!(failure.description.contains("status 7"), "{failure}");
}

#[test]
fn should_prefer_a_parked_closure_failure_over_the_status_description() {
    install_non_terminating_error_handler();
    set_closure_error(MlxCError {
        operation: "the root cause",
        description: "parked by a trampoline".to_owned(),
    });

    let failure = check_status(1, "the outer call").expect_err("nonzero is a failure");

    assert_eq!(failure.operation, "the root cause");
    assert_eq!(failure.description, "parked by a trampoline");
}

#[test]
fn should_discard_a_parked_failure_when_a_later_status_succeeds() {
    install_non_terminating_error_handler();
    set_closure_error(MlxCError {
        operation: "a stale failure",
        description: "must not resurface".to_owned(),
    });

    check_status(0, "a successful call").expect("zero is success");

    let failure = check_status(1, "a later failure").expect_err("nonzero is a failure");
    assert_eq!(failure.operation, "a later failure");
}

#[test]
fn should_clear_the_captured_description_when_it_is_taken() {
    let context = gpu_context();
    let values = float_array(&[1.0], &[1]);
    // `dimension` has no status to check, so its description stays parked.
    assert_eq!(values.dimension(9), 0);

    assert!(take_captured_mlx_error().is_some());
    assert!(take_captured_mlx_error().is_none());
    drop(context);
}

fn fresh_thread_has_no_captured_error() -> bool {
    take_captured_mlx_error().is_none()
}

#[test]
fn should_keep_captured_descriptions_on_the_thread_that_failed() {
    gpu_context();
    let values = float_array(&[1.0], &[1]);
    assert_eq!(values.dimension(9), 0);

    let other_thread = std::thread::spawn(fresh_thread_has_no_captured_error);

    assert!(
        other_thread.join().expect("the thread should finish"),
        "another thread must not observe this thread's failure"
    );
    assert!(take_captured_mlx_error().is_some());
}

#[test]
fn should_install_the_error_handler_idempotently() {
    install_non_terminating_error_handler();
    install_non_terminating_error_handler();
    let context = gpu_context();

    assert!(context.reshape(&float_array(&[1.0], &[1]), &[2]).is_err());
}
