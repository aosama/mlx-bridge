//! Tier 2 contracts for the closure trampolines, the crate's highest-risk ABI
//! surface: Rust callables crossing the C boundary, their failures returning
//! as typed errors, and panics stopping at the FFI edge.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{array_at, float_array, floats_of, gpu_context, vector_of};
use mlx_c_rust::error::take_closure_error;
use mlx_c_rust::{
    MlxArray, MlxBindingsContext, MlxCError, MlxClosure, MlxClosureKwargs, MlxClosureValueAndGrad,
    MlxMapStringToArray,
};

fn doubled_inputs(inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        outputs.push(context.multiply_scalar(input, 2.0)?);
    }
    Ok(outputs)
}

fn input_count_as_scalar(inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let input_count = i32::try_from(inputs.len()).expect("the fixture has few inputs");
    Ok(vec![MlxArray::from_scalar_i32(input_count)])
}

fn failing_callback(_inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    Err(MlxCError {
        operation: "run the fixture callback",
        description: "the callback refused".to_owned(),
    })
}

fn panicking_callback(_inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    panic!("the fixture callback panicked")
}

fn panicking_with_formatted_message(_inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let attempt_number = 7;
    panic!("attempt {attempt_number} panicked")
}

fn keyword_scaled_input(
    inputs: &[MlxArray],
    keywords: &MlxMapStringToArray,
) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    let scale = keywords.value_at("scale")?.ok_or(MlxCError {
        operation: "read the fixture keyword",
        description: "the scale keyword is required".to_owned(),
    })?;
    Ok(vec![context.multiply(&inputs[0], &scale)?])
}

fn panicking_keyword_callback(
    _inputs: &[MlxArray],
    _keywords: &MlxMapStringToArray,
) -> Result<Vec<MlxArray>, MlxCError> {
    panic!("the keyword callback panicked")
}

fn square_value_with_doubled_gradient(
    inputs: &[MlxArray],
) -> Result<(Vec<MlxArray>, Vec<MlxArray>), MlxCError> {
    let context = MlxBindingsContext::new()?;
    let squared = context.square(&inputs[0])?;
    let gradient = context.multiply_scalar(&inputs[0], 2.0)?;
    Ok((vec![squared], vec![gradient]))
}

fn failing_value_and_grad(
    _inputs: &[MlxArray],
) -> Result<(Vec<MlxArray>, Vec<MlxArray>), MlxCError> {
    Err(MlxCError {
        operation: "run the fixture gradient",
        description: "the gradient refused".to_owned(),
    })
}

fn panicking_value_and_grad(
    _inputs: &[MlxArray],
) -> Result<(Vec<MlxArray>, Vec<MlxArray>), MlxCError> {
    panic!("the gradient callback panicked")
}

#[test]
fn should_apply_a_rust_callable_through_the_c_trampoline() {
    gpu_context();
    let closure = MlxClosure::from_fn(doubled_inputs).expect("the closure should build");
    let input = float_array(&[1.0, 2.0], &[2]);

    let outputs = closure
        .apply(&vector_of(&[&input]))
        .expect("apply should work");

    assert_eq!(outputs.len(), 1);
    assert_eq!(floats_of(&array_at(&outputs, 0)), vec![2.0, 4.0]);
}

#[test]
fn should_pass_every_input_and_return_every_output_in_order() {
    gpu_context();
    let closure = MlxClosure::from_fn(doubled_inputs).expect("the closure should build");
    let first = float_array(&[1.0], &[1]);
    let second = float_array(&[10.0], &[1]);

    let outputs = closure
        .apply(&vector_of(&[&first, &second]))
        .expect("apply should work");

    assert_eq!(floats_of(&array_at(&outputs, 0)), vec![2.0]);
    assert_eq!(floats_of(&array_at(&outputs, 1)), vec![20.0]);
}

#[test]
fn should_apply_a_callable_to_an_empty_input_vector() {
    gpu_context();
    let closure = MlxClosure::from_fn(input_count_as_scalar).expect("the closure should build");

    let outputs = closure.apply(&vector_of(&[])).expect("apply should work");

    assert_eq!(array_at(&outputs, 0).item_i32().unwrap(), 0);
}

#[test]
fn should_keep_callable_outputs_readable_after_the_closure_drops() {
    gpu_context();
    let closure = MlxClosure::from_fn(doubled_inputs).expect("the closure should build");
    let input = float_array(&[3.0], &[1]);

    let outputs = closure
        .apply(&vector_of(&[&input]))
        .expect("apply should work");
    drop(closure);

    assert_eq!(floats_of(&array_at(&outputs, 0)), vec![6.0]);
}

#[test]
fn should_surface_a_callable_failure_as_its_own_typed_error() {
    gpu_context();
    let closure = MlxClosure::from_fn(failing_callback).expect("the closure should build");

    let failure = closure
        .apply(&vector_of(&[]))
        .expect_err("the callback's failure must cross the C boundary");

    assert_eq!(failure.operation, "run the fixture callback");
    assert_eq!(failure.description, "the callback refused");
}

#[test]
fn should_consume_the_parked_failure_when_it_is_surfaced() {
    gpu_context();
    let closure = MlxClosure::from_fn(failing_callback).expect("the closure should build");

    closure
        .apply(&vector_of(&[]))
        .expect_err("the callback should fail");

    assert!(
        take_closure_error().is_none(),
        "a surfaced failure must not leak into the next operation"
    );
}

#[test]
fn should_succeed_on_the_call_after_a_callable_failure() {
    gpu_context();
    let failing = MlxClosure::from_fn(failing_callback).expect("the closure should build");
    let working = MlxClosure::from_fn(doubled_inputs).expect("the closure should build");
    let input = float_array(&[1.0], &[1]);

    failing
        .apply(&vector_of(&[]))
        .expect_err("the callback should fail");

    let outputs = working
        .apply(&vector_of(&[&input]))
        .expect("apply should work");
    assert_eq!(floats_of(&array_at(&outputs, 0)), vec![2.0]);
}

#[test]
fn should_stop_a_panic_at_the_ffi_boundary_and_report_its_message() {
    gpu_context();
    let closure = MlxClosure::from_fn(panicking_callback).expect("the closure should build");

    let failure = closure
        .apply(&vector_of(&[]))
        .expect_err("a panic must become an error, not an unwind across C");

    assert_eq!(failure.operation, "apply a Rust closure");
    assert_eq!(failure.description, "the fixture callback panicked");
}

#[test]
fn should_report_a_formatted_panic_message() {
    gpu_context();
    let closure =
        MlxClosure::from_fn(panicking_with_formatted_message).expect("the closure should build");

    let failure = closure
        .apply(&vector_of(&[]))
        .expect_err("the panic should surface");

    assert_eq!(failure.description, "attempt 7 panicked");
}

#[test]
fn should_keep_a_closure_usable_after_a_panic_inside_it() {
    gpu_context();
    let closure = MlxClosure::from_fn(panicking_callback).expect("the closure should build");

    closure
        .apply(&vector_of(&[]))
        .expect_err("the first panic should surface");

    closure
        .apply(&vector_of(&[]))
        .expect_err("the second panic should surface too");
}

#[test]
fn should_fail_to_apply_an_empty_closure() {
    gpu_context();
    let closure = MlxClosure::empty();

    let failure = closure
        .apply(&vector_of(&[]))
        .expect_err("an empty closure holds no callable");

    assert_eq!(failure.operation, "apply an MLX closure");
    assert!(failure.description.contains("non-empty"), "{failure}");
}

#[test]
fn should_share_the_callable_through_a_copied_closure_handle() {
    gpu_context();
    let source = MlxClosure::from_fn(doubled_inputs).expect("the closure should build");
    let mut copy = MlxClosure::empty();
    let input = float_array(&[2.0], &[1]);

    copy.set(&source).expect("the copy should work");
    drop(source);

    let outputs = copy
        .apply(&vector_of(&[&input]))
        .expect("apply should work");
    assert_eq!(floats_of(&array_at(&outputs, 0)), vec![4.0]);
}

#[test]
fn should_pass_keyword_arguments_and_leave_the_caller_map_intact() {
    gpu_context();
    let closure = MlxClosureKwargs::from_fn(keyword_scaled_input).expect("should build");
    let input = float_array(&[2.0], &[1]);
    let mut keywords = MlxMapStringToArray::new().expect("the map should build");
    keywords
        .insert("scale", &float_array(&[3.0], &[1]))
        .unwrap();

    let outputs = closure
        .apply(&vector_of(&[&input]), &keywords)
        .expect("apply should work");

    assert_eq!(floats_of(&array_at(&outputs, 0)), vec![6.0]);
    assert!(
        keywords.value_at("scale").unwrap().is_some(),
        "the trampoline only borrows the keyword map"
    );
}

#[test]
fn should_apply_a_keyword_closure_repeatedly() {
    gpu_context();
    let closure = MlxClosureKwargs::from_fn(keyword_scaled_input).expect("should build");
    let input = float_array(&[1.0], &[1]);
    let mut keywords = MlxMapStringToArray::new().expect("the map should build");
    keywords
        .insert("scale", &float_array(&[5.0], &[1]))
        .unwrap();

    for _application_number in 0..3 {
        let outputs = closure
            .apply(&vector_of(&[&input]), &keywords)
            .expect("apply should work");
        assert_eq!(floats_of(&array_at(&outputs, 0)), vec![5.0]);
    }
}

#[test]
fn should_surface_a_keyword_callback_failure_for_a_missing_keyword() {
    gpu_context();
    let closure = MlxClosureKwargs::from_fn(keyword_scaled_input).expect("should build");
    let input = float_array(&[2.0], &[1]);
    let keywords = MlxMapStringToArray::new().expect("the map should build");

    let failure = closure
        .apply(&vector_of(&[&input]), &keywords)
        .expect_err("the callback requires the keyword");

    assert_eq!(failure.operation, "read the fixture keyword");
}

#[test]
fn should_stop_a_keyword_callback_panic_at_the_ffi_boundary() {
    gpu_context();
    let closure = MlxClosureKwargs::from_fn(panicking_keyword_callback).expect("should build");
    let keywords = MlxMapStringToArray::new().expect("the map should build");

    let failure = closure
        .apply(&vector_of(&[]), &keywords)
        .expect_err("the panic should surface");

    assert_eq!(failure.description, "the keyword callback panicked");
}

#[test]
fn should_fail_to_apply_an_empty_keyword_closure() {
    gpu_context();
    let keywords = MlxMapStringToArray::new().expect("the map should build");

    assert!(
        MlxClosureKwargs::empty()
            .apply(&vector_of(&[]), &keywords)
            .is_err()
    );
}

#[test]
fn should_return_both_values_and_gradients_from_a_value_and_grad_callable() {
    gpu_context();
    let closure =
        MlxClosureValueAndGrad::from_fn(square_value_with_doubled_gradient).expect("should build");
    let input = float_array(&[3.0], &[1]);

    let (values, gradients) = closure
        .apply(&vector_of(&[&input]))
        .expect("apply should work");

    assert_eq!(floats_of(&array_at(&values, 0)), vec![9.0]);
    assert_eq!(floats_of(&array_at(&gradients, 0)), vec![6.0]);
}

#[test]
fn should_surface_a_value_and_grad_callback_failure() {
    gpu_context();
    let closure = MlxClosureValueAndGrad::from_fn(failing_value_and_grad).expect("should build");

    let failure = closure
        .apply(&vector_of(&[]))
        .expect_err("the callback should fail");

    assert_eq!(failure.operation, "run the fixture gradient");
}

#[test]
fn should_stop_a_value_and_grad_panic_at_the_ffi_boundary() {
    gpu_context();
    let closure = MlxClosureValueAndGrad::from_fn(panicking_value_and_grad).expect("should build");

    let failure = closure
        .apply(&vector_of(&[]))
        .expect_err("the panic should surface");

    assert_eq!(failure.description, "the gradient callback panicked");
}

#[test]
fn should_fail_to_apply_an_empty_value_and_grad_closure() {
    gpu_context();

    assert!(
        MlxClosureValueAndGrad::empty()
            .apply(&vector_of(&[]))
            .is_err()
    );
}
