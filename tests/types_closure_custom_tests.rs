//! Tier 2 contracts for the custom-transform closure families, applied
//! directly through their own trampolines.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{array_at, float_array, floats_of, gpu_context, vector_of};
use mlx_bridge::{
    MlxArray, MlxCError, MlxClosureCustom, MlxClosureCustomJvp, MlxClosureCustomVmap,
};

fn counted_arguments(
    primals: &[MlxArray],
    outputs: &[MlxArray],
    cotangents: &[MlxArray],
) -> Result<Vec<MlxArray>, MlxCError> {
    let argument_counts = [
        primals.len() as f32,
        outputs.len() as f32,
        cotangents.len() as f32,
    ];
    Ok(vec![MlxArray::from_f32(&argument_counts, &[3])?])
}

fn refusing_custom_rule(
    _primals: &[MlxArray],
    _outputs: &[MlxArray],
    _cotangents: &[MlxArray],
) -> Result<Vec<MlxArray>, MlxCError> {
    Err(MlxCError {
        operation: "run the custom rule",
        description: "the rule refused".to_owned(),
    })
}

fn panicking_custom_rule(
    _primals: &[MlxArray],
    _outputs: &[MlxArray],
    _cotangents: &[MlxArray],
) -> Result<Vec<MlxArray>, MlxCError> {
    panic!("the custom rule panicked")
}

fn output_indices_as_floats(
    _primals: &[MlxArray],
    _tangents: &[MlxArray],
    output_indices: &[i32],
) -> Result<Vec<MlxArray>, MlxCError> {
    let mut index_values = Vec::with_capacity(output_indices.len());
    for output_index in output_indices {
        index_values.push(*output_index as f32);
    }
    let value_count = i32::try_from(index_values.len()).expect("the fixture has few indices");
    Ok(vec![MlxArray::from_f32(&index_values, &[value_count])?])
}

fn refusing_jvp_rule(
    _primals: &[MlxArray],
    _tangents: &[MlxArray],
    _output_indices: &[i32],
) -> Result<Vec<MlxArray>, MlxCError> {
    Err(MlxCError {
        operation: "run the JVP rule",
        description: "the rule refused".to_owned(),
    })
}

fn panicking_jvp_rule(
    _primals: &[MlxArray],
    _tangents: &[MlxArray],
    _output_indices: &[i32],
) -> Result<Vec<MlxArray>, MlxCError> {
    panic!("the JVP rule panicked")
}

fn shifted_batch_axes(
    inputs: &[MlxArray],
    input_axes: &[i32],
) -> Result<(Vec<MlxArray>, Vec<i32>), MlxCError> {
    let mut shifted_axes = Vec::with_capacity(input_axes.len());
    for input_axis in input_axes {
        shifted_axes.push(input_axis + 1);
    }
    Ok((vec![inputs[0].retain()?], shifted_axes))
}

fn refusing_vmap_rule(
    _inputs: &[MlxArray],
    _input_axes: &[i32],
) -> Result<(Vec<MlxArray>, Vec<i32>), MlxCError> {
    Err(MlxCError {
        operation: "run the VMAP rule",
        description: "the rule refused".to_owned(),
    })
}

fn panicking_vmap_rule(
    _inputs: &[MlxArray],
    _input_axes: &[i32],
) -> Result<(Vec<MlxArray>, Vec<i32>), MlxCError> {
    panic!("the VMAP rule panicked")
}

#[test]
fn should_pass_all_three_argument_groups_to_a_custom_rule() {
    gpu_context();
    let rule = MlxClosureCustom::from_fn(counted_arguments).expect("the rule should build");
    let array = float_array(&[1.0], &[1]);

    let outputs = rule
        .apply(
            &vector_of(&[&array]),
            &vector_of(&[&array, &array]),
            &vector_of(&[&array, &array, &array]),
        )
        .expect("apply should work");

    assert_eq!(floats_of(&array_at(&outputs, 0)), vec![1.0, 2.0, 3.0]);
}

#[test]
fn should_surface_a_custom_rule_failure() {
    gpu_context();
    let rule = MlxClosureCustom::from_fn(refusing_custom_rule).expect("the rule should build");

    let failure = rule
        .apply(&vector_of(&[]), &vector_of(&[]), &vector_of(&[]))
        .expect_err("the rule should fail");

    assert_eq!(failure.operation, "run the custom rule");
}

#[test]
fn should_stop_a_custom_rule_panic_at_the_ffi_boundary() {
    gpu_context();
    let rule = MlxClosureCustom::from_fn(panicking_custom_rule).expect("the rule should build");

    let failure = rule
        .apply(&vector_of(&[]), &vector_of(&[]), &vector_of(&[]))
        .expect_err("the panic should surface");

    assert_eq!(failure.description, "the custom rule panicked");
}

#[test]
fn should_fail_to_apply_an_empty_custom_rule() {
    gpu_context();

    assert!(
        MlxClosureCustom::empty()
            .apply(&vector_of(&[]), &vector_of(&[]), &vector_of(&[]))
            .is_err()
    );
}

#[test]
fn should_share_a_custom_rule_through_a_copied_handle() {
    gpu_context();
    let source = MlxClosureCustom::from_fn(counted_arguments).expect("the rule should build");
    let mut copy = MlxClosureCustom::empty();

    copy.set(&source).expect("the copy should work");
    drop(source);

    let outputs = copy
        .apply(&vector_of(&[]), &vector_of(&[]), &vector_of(&[]))
        .expect("apply should work");
    assert_eq!(floats_of(&array_at(&outputs, 0)), vec![0.0, 0.0, 0.0]);
}

#[test]
fn should_pass_output_indices_to_a_jvp_rule() {
    gpu_context();
    let rule = MlxClosureCustomJvp::from_fn(output_indices_as_floats).expect("should build");

    let outputs = rule
        .apply(&vector_of(&[]), &vector_of(&[]), &[2, 5])
        .expect("apply should work");

    assert_eq!(floats_of(&array_at(&outputs, 0)), vec![2.0, 5.0]);
}

#[test]
fn should_pass_an_empty_index_list_to_a_jvp_rule() {
    gpu_context();
    let rule = MlxClosureCustomJvp::from_fn(output_indices_as_floats).expect("should build");

    let outputs = rule
        .apply(&vector_of(&[]), &vector_of(&[]), &[])
        .expect("apply should work");

    assert_eq!(floats_of(&array_at(&outputs, 0)), Vec::<f32>::new());
}

#[test]
fn should_surface_a_jvp_rule_failure_and_stop_its_panic() {
    gpu_context();
    let refusing = MlxClosureCustomJvp::from_fn(refusing_jvp_rule).expect("should build");
    let panicking = MlxClosureCustomJvp::from_fn(panicking_jvp_rule).expect("should build");

    let failure = refusing
        .apply(&vector_of(&[]), &vector_of(&[]), &[])
        .expect_err("the rule should fail");
    let panic_failure = panicking
        .apply(&vector_of(&[]), &vector_of(&[]), &[])
        .expect_err("the panic should surface");

    assert_eq!(failure.operation, "run the JVP rule");
    assert_eq!(panic_failure.description, "the JVP rule panicked");
}

#[test]
fn should_fail_to_apply_an_empty_jvp_rule() {
    gpu_context();

    assert!(
        MlxClosureCustomJvp::empty()
            .apply(&vector_of(&[]), &vector_of(&[]), &[])
            .is_err()
    );
}

#[test]
fn should_return_arrays_and_axes_from_a_vmap_rule() {
    gpu_context();
    let rule = MlxClosureCustomVmap::from_fn(shifted_batch_axes).expect("should build");
    let array = float_array(&[1.0], &[1]);

    let (outputs, output_axes) = rule
        .apply(&vector_of(&[&array]), &[0])
        .expect("apply should work");

    assert_eq!(outputs.len(), 1);
    assert_eq!(output_axes.value_at(0).unwrap(), 1);
}

#[test]
fn should_surface_a_vmap_rule_failure_and_stop_its_panic() {
    gpu_context();
    let refusing = MlxClosureCustomVmap::from_fn(refusing_vmap_rule).expect("should build");
    let panicking = MlxClosureCustomVmap::from_fn(panicking_vmap_rule).expect("should build");

    let failure = refusing
        .apply(&vector_of(&[]), &[])
        .expect_err("the rule should fail");
    let panic_failure = panicking
        .apply(&vector_of(&[]), &[])
        .expect_err("the panic should surface");

    assert_eq!(failure.operation, "run the VMAP rule");
    assert_eq!(panic_failure.description, "the VMAP rule panicked");
}

#[test]
fn should_fail_to_apply_an_empty_vmap_rule() {
    gpu_context();

    assert!(
        MlxClosureCustomVmap::empty()
            .apply(&vector_of(&[]), &[])
            .is_err()
    );
}
