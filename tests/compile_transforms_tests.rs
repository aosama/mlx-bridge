//! Tier 2 and Tier 7 contracts for the transforms: differentiation and batch
//! mapping through Rust callables, custom rules, and compiled replays that
//! must agree with the eager path.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{
    FLOAT_TOLERANCE, array_at, assert_floats_close, float_array, floats_of, gpu_context, vector_of,
};
use mlx_c_rust::{
    MlxArray, MlxBindingsContext, MlxCError, MlxClosure, MlxClosureCustom, MlxClosureCustomJvp,
    MlxClosureCustomVmap, MlxCompileCache, disable_compile, enable_compile, jvp_of_closure, raw,
    set_compile_mode, value_and_grad_of_closure, vjp_of_closure, vmap_replace, vmap_trace,
};

fn sum_of_squares(inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    Ok(vec![context.sum(&context.square(&inputs[0])?, false)?])
}

fn sum_of_products(inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    Ok(vec![
        context.sum(&context.multiply(&inputs[0], &inputs[1])?, false)?,
    ])
}

fn doubled(inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    Ok(vec![context.multiply_scalar(&inputs[0], 2.0)?])
}

fn refusing_callback(_inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    Err(MlxCError {
        operation: "run the traced fixture",
        description: "tracing refused".to_owned(),
    })
}

fn panicking_callback(_inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    panic!("the traced fixture panicked")
}

fn constant_gradient_rule(
    _primals: &[MlxArray],
    _outputs: &[MlxArray],
    _cotangents: &[MlxArray],
) -> Result<Vec<MlxArray>, MlxCError> {
    Ok(vec![MlxArray::from_scalar_f32(42.0)])
}

fn constant_tangent_rule(
    _primals: &[MlxArray],
    _tangents: &[MlxArray],
    _output_indices: &[i32],
) -> Result<Vec<MlxArray>, MlxCError> {
    Ok(vec![MlxArray::from_scalar_f32(11.0)])
}

fn identity_batch_rule(
    inputs: &[MlxArray],
    input_axes: &[i32],
) -> Result<(Vec<MlxArray>, Vec<i32>), MlxCError> {
    Ok((vec![inputs[0].retain()?], input_axes.to_vec()))
}

#[test]
fn should_differentiate_a_square_at_a_known_point() {
    gpu_context();
    let function = MlxClosure::from_fn(sum_of_squares).expect("the closure should build");
    let value_and_grad = value_and_grad_of_closure(&function, &[0]).expect("should build");
    let point = float_array(&[3.0], &[1]);

    let (values, gradients) = value_and_grad
        .apply(&vector_of(&[&point]))
        .expect("the transform should apply");

    assert_eq!(array_at(&values, 0).item_f32().unwrap(), 9.0);
    assert_eq!(floats_of(&array_at(&gradients, 0)), vec![6.0]);
}

#[test]
fn should_differentiate_only_the_selected_argument() {
    gpu_context();
    let function = MlxClosure::from_fn(sum_of_products).expect("the closure should build");
    let value_and_grad = value_and_grad_of_closure(&function, &[1]).expect("should build");
    let left = float_array(&[2.0], &[1]);
    let right = float_array(&[5.0], &[1]);

    let (_values, gradients) = value_and_grad
        .apply(&vector_of(&[&left, &right]))
        .expect("the transform should apply");

    assert_eq!(gradients.len(), 1);
    assert_eq!(floats_of(&array_at(&gradients, 0)), vec![2.0]);
}

#[test]
fn should_refuse_a_gradient_over_no_arguments() {
    gpu_context();
    let function = MlxClosure::from_fn(sum_of_squares).expect("the closure should build");
    let point = float_array(&[3.0], &[1]);

    let applied_without_failure = match value_and_grad_of_closure(&function, &[]) {
        Ok(value_and_grad) => value_and_grad.apply(&vector_of(&[&point])).is_ok(),
        Err(_build_failure) => false,
    };

    assert!(!applied_without_failure);
}

#[test]
fn should_surface_a_callback_failure_from_inside_a_gradient_trace() {
    gpu_context();
    let function = MlxClosure::from_fn(refusing_callback).expect("the closure should build");
    let value_and_grad = value_and_grad_of_closure(&function, &[0]).expect("should build");
    let point = float_array(&[3.0], &[1]);

    let failure = value_and_grad
        .apply(&vector_of(&[&point]))
        .expect_err("the trace must report the callback's failure");

    assert!(!failure.description.is_empty());
}

#[test]
fn should_stop_a_panic_from_inside_a_gradient_trace() {
    gpu_context();
    let function = MlxClosure::from_fn(panicking_callback).expect("the closure should build");
    let value_and_grad = value_and_grad_of_closure(&function, &[0]).expect("should build");
    let point = float_array(&[3.0], &[1]);

    assert!(value_and_grad.apply(&vector_of(&[&point])).is_err());
}

#[test]
fn should_compute_a_vector_jacobian_product() {
    gpu_context();
    let function = MlxClosure::from_fn(sum_of_squares).expect("the closure should build");
    let point = float_array(&[3.0], &[1]);
    let cotangent = MlxArray::from_scalar_f32(1.0);

    let (outputs, cotangents) =
        vjp_of_closure(&function, &vector_of(&[&point]), &vector_of(&[&cotangent]))
            .expect("the transform should apply");

    assert_eq!(array_at(&outputs, 0).item_f32().unwrap(), 9.0);
    assert_eq!(floats_of(&array_at(&cotangents, 0)), vec![6.0]);
}

#[test]
fn should_compute_a_jacobian_vector_product() {
    gpu_context();
    let function = MlxClosure::from_fn(sum_of_squares).expect("the closure should build");
    let point = float_array(&[3.0], &[1]);
    let tangent = float_array(&[1.0], &[1]);

    let (outputs, tangents) =
        jvp_of_closure(&function, &vector_of(&[&point]), &vector_of(&[&tangent]))
            .expect("the transform should apply");

    assert_eq!(array_at(&outputs, 0).item_f32().unwrap(), 9.0);
    assert_eq!(array_at(&tangents, 0).item_f32().unwrap(), 6.0);
}

#[test]
fn should_map_a_callable_over_the_batch_axis() {
    gpu_context();
    let function = MlxClosure::from_fn(doubled).expect("the closure should build");
    let batch = float_array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let inputs = vector_of(&[&batch]);

    let (traced_inputs, traced_outputs, _output_axes) =
        vmap_trace(&function, &inputs, &[0]).expect("the trace should work");
    let mapped_outputs = vmap_replace(&inputs, &traced_inputs, &traced_outputs, &[0], &[0])
        .expect("the replacement should work");

    let mapped = array_at(&mapped_outputs, 0);
    assert_eq!(mapped.shape(), vec![2, 2]);
    assert_eq!(floats_of(&mapped), vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn should_place_the_mapped_axis_where_the_output_axes_ask() {
    gpu_context();
    let function = MlxClosure::from_fn(doubled).expect("the closure should build");
    let batch = float_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let inputs = vector_of(&[&batch]);

    let (traced_inputs, traced_outputs, _output_axes) =
        vmap_trace(&function, &inputs, &[0]).expect("the trace should work");
    let mapped_outputs = vmap_replace(&inputs, &traced_inputs, &traced_outputs, &[0], &[1])
        .expect("the replacement should work");

    assert_eq!(array_at(&mapped_outputs, 0).shape(), vec![3, 2]);
}

#[test]
fn should_match_the_eager_path_through_a_compiled_closure() {
    gpu_context();
    let eager = MlxClosure::from_fn(doubled).expect("the closure should build");
    let compiled = eager.compiled(false).expect("the closure should compile");
    let input = float_array(&[1.5, -2.0], &[2]);

    let eager_outputs = eager
        .apply(&vector_of(&[&input]))
        .expect("apply should work");
    let compiled_outputs = compiled
        .apply(&vector_of(&[&input]))
        .expect("apply should work");

    assert_eq!(
        floats_of(&array_at(&eager_outputs, 0)),
        floats_of(&array_at(&compiled_outputs, 0))
    );
}

#[test]
fn should_replay_a_shapeless_compilation_across_input_shapes() {
    gpu_context();
    let eager = MlxClosure::from_fn(doubled).expect("the closure should build");
    let compiled = eager.compiled(true).expect("the closure should compile");
    let short_input = float_array(&[1.0], &[1]);
    let long_input = float_array(&[1.0, 2.0, 3.0], &[3]);

    let short_outputs = compiled.apply(&vector_of(&[&short_input])).unwrap();
    let long_outputs = compiled.apply(&vector_of(&[&long_input])).unwrap();

    assert_eq!(floats_of(&array_at(&short_outputs, 0)), vec![2.0]);
    assert_eq!(floats_of(&array_at(&long_outputs, 0)), vec![2.0, 4.0, 6.0]);
}

#[test]
fn should_keep_a_compiled_closure_usable_after_its_source_drops() {
    gpu_context();
    let eager = MlxClosure::from_fn(doubled).expect("the closure should build");
    let compiled = eager.compiled(false).expect("the closure should compile");
    let input = float_array(&[4.0], &[1]);

    drop(eager);

    let outputs = compiled
        .apply(&vector_of(&[&input]))
        .expect("apply should work");
    assert_eq!(floats_of(&array_at(&outputs, 0)), vec![8.0]);
}

#[test]
fn should_surface_a_callback_failure_from_a_compiled_trace() {
    gpu_context();
    let refusing = MlxClosure::from_fn(refusing_callback).expect("the closure should build");
    let compiled = refusing.compiled(false).expect("building is lazy");
    let input = float_array(&[1.0], &[1]);

    assert!(compiled.apply(&vector_of(&[&input])).is_err());
}

#[test]
fn should_refuse_to_compile_an_empty_closure() {
    gpu_context();
    let empty = MlxClosure::empty();

    assert!(empty.compiled(false).is_err());
    assert!(empty.checkpointed().is_err());
}

#[test]
fn should_compute_the_same_result_across_every_compile_mode() {
    gpu_context();
    let eager = MlxClosure::from_fn(doubled).expect("the closure should build");
    let input = float_array(&[2.0], &[1]);
    let compile_modes = [
        raw::mlx_compile_mode__MLX_COMPILE_MODE_DISABLED,
        raw::mlx_compile_mode__MLX_COMPILE_MODE_NO_SIMPLIFY,
        raw::mlx_compile_mode__MLX_COMPILE_MODE_NO_FUSE,
        raw::mlx_compile_mode__MLX_COMPILE_MODE_ENABLED,
    ];

    for compile_mode in compile_modes {
        set_compile_mode(compile_mode).expect("the mode should apply");
        let compiled = eager.compiled(false).expect("the closure should compile");
        let outputs = compiled
            .apply(&vector_of(&[&input]))
            .expect("apply should work");
        assert_eq!(floats_of(&array_at(&outputs, 0)), vec![4.0]);
    }
}

#[test]
fn should_compute_the_same_result_with_compilation_disabled_and_reenabled() {
    gpu_context();
    let eager = MlxClosure::from_fn(doubled).expect("the closure should build");
    let input = float_array(&[3.0], &[1]);

    disable_compile().expect("disabling should work");
    let while_disabled = eager
        .compiled(false)
        .unwrap()
        .apply(&vector_of(&[&input]))
        .unwrap();
    enable_compile().expect("enabling should work");
    let while_enabled = eager
        .compiled(false)
        .unwrap()
        .apply(&vector_of(&[&input]))
        .unwrap();

    assert_eq!(floats_of(&array_at(&while_disabled, 0)), vec![6.0]);
    assert_eq!(floats_of(&array_at(&while_enabled, 0)), vec![6.0]);
}

#[test]
fn should_keep_values_and_gradients_through_a_checkpoint() {
    gpu_context();
    let function = MlxClosure::from_fn(sum_of_squares).expect("the closure should build");
    let checkpointed = function
        .checkpointed()
        .expect("the checkpoint should build");
    let value_and_grad = value_and_grad_of_closure(&checkpointed, &[0]).expect("should build");
    let point = float_array(&[2.0], &[1]);

    let (_values, gradients) = value_and_grad.apply(&vector_of(&[&point])).unwrap();

    assert_floats_close(
        &floats_of(&array_at(&gradients, 0)),
        &[4.0],
        FLOAT_TOLERANCE,
    );
}

#[test]
fn should_replace_the_gradient_through_a_custom_vjp() {
    gpu_context();
    let function = MlxClosure::from_fn(sum_of_squares).expect("the closure should build");
    let backward = MlxClosureCustom::from_fn(constant_gradient_rule).expect("should build");
    let customized = function.with_custom_vjp(&backward).expect("should attach");
    let value_and_grad = value_and_grad_of_closure(&customized, &[0]).expect("should build");
    let point = float_array(&[2.0], &[1]);

    let (values, gradients) = value_and_grad.apply(&vector_of(&[&point])).unwrap();

    assert_eq!(array_at(&values, 0).item_f32().unwrap(), 4.0);
    assert_eq!(array_at(&gradients, 0).item_f32().unwrap(), 42.0);
}

#[test]
fn should_apply_every_custom_rule_attached_to_a_closure() {
    gpu_context();
    let function = MlxClosure::from_fn(sum_of_squares).expect("the closure should build");
    let backward = MlxClosureCustom::from_fn(constant_gradient_rule).expect("should build");
    let forward = MlxClosureCustomJvp::from_fn(constant_tangent_rule).expect("should build");
    let batched = MlxClosureCustomVmap::from_fn(identity_batch_rule).expect("should build");
    let customized = function
        .with_custom_rules(&backward, &forward, &batched)
        .expect("should attach");
    let point = float_array(&[2.0], &[1]);
    let tangent = float_array(&[1.0], &[1]);

    let (_outputs, tangents) =
        jvp_of_closure(&customized, &vector_of(&[&point]), &vector_of(&[&tangent])).unwrap();
    let value_and_grad = value_and_grad_of_closure(&customized, &[0]).expect("should build");
    let (_values, gradients) = value_and_grad.apply(&vector_of(&[&point])).unwrap();

    assert_eq!(array_at(&tangents, 0).item_f32().unwrap(), 11.0);
    assert_eq!(array_at(&gradients, 0).item_f32().unwrap(), 42.0);
}

#[test]
fn should_compile_through_the_detailed_entry_and_clear_its_cache_entry() {
    gpu_context();
    const FUNCTION_IDENTITY: usize = 0x5EED;
    let eager = MlxClosure::from_fn(doubled).expect("the closure should build");
    let compiled = eager
        .compile_detailed(FUNCTION_IDENTITY, false, &[])
        .expect("the closure should compile");
    let input = float_array(&[1.0], &[1]);

    let outputs = compiled
        .apply(&vector_of(&[&input]))
        .expect("apply should work");
    let mut cache = MlxCompileCache::empty();

    assert_eq!(floats_of(&array_at(&outputs, 0)), vec![2.0]);
    cache
        .erase(FUNCTION_IDENTITY)
        .expect("the erase should work");
    cache.clear().expect("the clear should work");
}

#[test]
fn should_keep_values_when_evaluating_after_dependencies() {
    gpu_context();
    let value = float_array(&[1.0, 2.0], &[2]);
    let dependency = float_array(&[3.0], &[1]);

    let outputs = value
        .evaluate_after(&vector_of(&[&dependency]))
        .expect("the dependency should schedule");

    assert_eq!(floats_of(&array_at(&outputs, 0)), vec![1.0, 2.0]);
}
