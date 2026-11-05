//! Tier 7 contracts for the prebuilt compiled-graph family: each replay must
//! match the eager operations it compiles, across the shapes it promises to
//! accept, and report failures like any other operation.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use std::os::raw::c_int;

use bridge_fixtures::{FLOAT_TOLERANCE, assert_floats_close, float_array, floats_of, gpu_context};
use mlx_c_rust::{
    MlxArray, MlxCompiledDualHeadNorm, MlxCompiledElementwiseGraphs,
    MlxCompiledGatedDeltaRateActivation, MlxCompiledMultiOutputGraph, MlxCompiledSwiGlu, MlxDtype,
    MlxStream, array_from_vector, raw, set_graph_output_vector,
};

/// Half precision cannot hold the float32 comparison tolerance.
const HALF_PRECISION_TOLERANCE: f32 = 1e-2;

/// A graph builder following the MLX closure ABI: outputs the sum and the
/// difference of its two inputs on the GPU stream.
unsafe extern "C" fn sum_and_difference_graph(
    output_vector: *mut raw::mlx_vector_array,
    input_vector: raw::mlx_vector_array,
) -> c_int {
    let Ok(left) = array_from_vector(input_vector, 0) else {
        return 1;
    };
    let Ok(right) = array_from_vector(input_vector, 1) else {
        return 1;
    };
    let Ok(gpu_stream) = MlxStream::default_gpu() else {
        return 1;
    };
    let mut sum = MlxArray::empty();
    let mut difference = MlxArray::empty();
    // SAFETY: Every input handle and the stream are live, and each output
    // handle is uniquely writable.
    let build_status = unsafe {
        raw::mlx_add(sum.raw_mut(), left.raw(), right.raw(), gpu_stream.raw())
            + raw::mlx_subtract(
                difference.raw_mut(),
                left.raw(),
                right.raw(),
                gpu_stream.raw(),
            )
    };
    if build_status != 0 {
        return 1;
    }
    // SAFETY: MLX provides a writable output vector for this callback.
    unsafe { set_graph_output_vector(output_vector, &[&sum, &difference]) }
}

#[test]
fn should_match_the_eager_path_through_the_compiled_swiglu() {
    let context = gpu_context();
    let compiled = MlxCompiledSwiGlu::new().expect("the graph should compile");
    let gate = float_array(&[-1.0, 0.0, 2.0], &[3]);
    let input = float_array(&[3.0, 4.0, 5.0], &[3]);

    let replayed = context
        .apply_compiled_swiglu(&compiled, &gate, &input)
        .unwrap();
    let eager = context
        .multiply(&context.silu(&gate).unwrap(), &input)
        .unwrap();

    assert_floats_close(&floats_of(&replayed), &floats_of(&eager), FLOAT_TOLERANCE);
}

#[test]
fn should_replay_the_shapeless_swiglu_across_input_shapes() {
    let context = gpu_context();
    let compiled = MlxCompiledSwiGlu::new().expect("the graph should compile");
    let short_gate = float_array(&[0.0], &[1]);
    let square_gate = float_array(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);

    let short_replay = context
        .apply_compiled_swiglu(&compiled, &short_gate, &short_gate)
        .unwrap();
    let square_replay = context
        .apply_compiled_swiglu(&compiled, &square_gate, &square_gate)
        .unwrap();

    assert_eq!(short_replay.shape(), vec![1]);
    assert_eq!(square_replay.shape(), vec![2, 2]);
}

#[test]
fn should_translate_a_compiled_replay_shape_mismatch() {
    let context = gpu_context();
    let compiled = MlxCompiledSwiGlu::new().expect("the graph should compile");
    let gate = float_array(&[1.0, 2.0], &[2]);
    let input = float_array(&[1.0, 2.0, 3.0], &[3]);

    assert!(
        context
            .apply_compiled_swiglu(&compiled, &gate, &input)
            .is_err()
    );
}

#[test]
fn should_match_the_eager_silu() {
    let context = gpu_context();
    let compiled = MlxCompiledElementwiseGraphs::new().expect("the graphs should compile");
    let input = float_array(&[-2.0, 0.0, 3.0], &[3]);

    let replayed = context.apply_compiled_silu(&compiled, &input).unwrap();

    assert_floats_close(
        &floats_of(&replayed),
        &floats_of(&context.silu(&input).unwrap()),
        FLOAT_TOLERANCE,
    );
}

#[test]
fn should_compute_the_documented_beta_scaled_softplus() {
    let context = gpu_context();
    let compiled = MlxCompiledElementwiseGraphs::new().expect("the graphs should compile");
    let input = float_array(&[-2.0, 0.0, 3.0], &[3]);
    let natural_log_of_two = std::f32::consts::LN_2;

    let replayed = context.apply_compiled_softplus(&compiled, &input).unwrap();
    // The documented formula, log1p(exp(input * ln2)) / ln2, deliberately
    // differs from the standard softplus.
    let scaled_input = context.multiply_scalar(&input, natural_log_of_two).unwrap();
    let scaled_softplus = context.log1p(&context.exp(&scaled_input).unwrap()).unwrap();
    let eager = context
        .multiply_scalar(&scaled_softplus, 1.0 / natural_log_of_two)
        .unwrap();

    assert_floats_close(&floats_of(&replayed), &floats_of(&eager), FLOAT_TOLERANCE);
}

#[test]
fn should_match_the_eager_attention_output_gate() {
    let context = gpu_context();
    let compiled = MlxCompiledElementwiseGraphs::new().expect("the graphs should compile");
    let attention_output = float_array(&[1.0, -2.0], &[2]);
    let gate_logits = float_array(&[0.5, -0.5], &[2]);

    let replayed = context
        .apply_compiled_attention_output_gate(&compiled, &attention_output, &gate_logits)
        .unwrap();
    let eager = context
        .multiply(&attention_output, &context.sigmoid(&gate_logits).unwrap())
        .unwrap();

    assert_floats_close(&floats_of(&replayed), &floats_of(&eager), FLOAT_TOLERANCE);
}

#[test]
fn should_match_two_eager_norms_through_the_dual_head_fusion() {
    let context = gpu_context();
    const NORM_EPSILON: f32 = 1e-5;
    let compiled = MlxCompiledDualHeadNorm::new(NORM_EPSILON).expect("the graph should compile");
    let queries = float_array(&[1.0, 2.0], &[1, 2]);
    let keys = float_array(&[3.0, 4.0], &[1, 2]);
    let query_weight = float_array(&[1.0, 1.0], &[2]);
    let key_weight = float_array(&[2.0, 2.0], &[2]);

    let (fused_queries, fused_keys) = context
        .apply_compiled_dual_head_norm(&compiled, &queries, &keys, &query_weight, &key_weight)
        .unwrap();
    let eager_queries = context
        .rms_norm(&queries, &query_weight, NORM_EPSILON)
        .unwrap();
    let eager_keys = context.rms_norm(&keys, &key_weight, NORM_EPSILON).unwrap();

    assert_floats_close(
        &floats_of(&fused_queries),
        &floats_of(&eager_queries),
        FLOAT_TOLERANCE,
    );
    assert_floats_close(
        &floats_of(&fused_keys),
        &floats_of(&eager_keys),
        FLOAT_TOLERANCE,
    );
}

#[test]
fn should_return_every_output_of_a_custom_multi_output_graph_in_order() {
    let context = gpu_context();
    let compiled = MlxCompiledMultiOutputGraph::new(
        sum_and_difference_graph,
        "compile the sum and difference fixture",
        true,
    )
    .expect("the graph should compile");
    let left = float_array(&[5.0], &[1]);
    let right = float_array(&[3.0], &[1]);

    let outputs = context
        .apply_compiled_multi_output_graph(&compiled, &[&left, &right])
        .unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(floats_of(&outputs[0]), vec![8.0]);
    assert_eq!(floats_of(&outputs[1]), vec![2.0]);
}

#[test]
fn should_report_a_graph_builder_that_cannot_read_its_inputs() {
    let context = gpu_context();
    let compiled = MlxCompiledMultiOutputGraph::new(
        sum_and_difference_graph,
        "compile the sum and difference fixture",
        true,
    )
    .expect("the graph should compile");
    let left = float_array(&[5.0], &[1]);

    assert!(
        context
            .apply_compiled_multi_output_graph(&compiled, &[&left])
            .is_err()
    );
}

#[test]
fn should_compute_the_documented_sparse_shared_expert_combination() {
    let context = gpu_context();
    let compiled = MlxCompiledElementwiseGraphs::new().expect("the graphs should compile");
    let sparse = float_array(&[1.0, 2.0], &[2]);
    let shared = float_array(&[4.0, 8.0], &[2]);
    let gate_logits = float_array(&[0.0, 0.0], &[2]);

    let replayed = context
        .apply_compiled_sparse_shared_expert_combination(&compiled, &sparse, &shared, &gate_logits)
        .unwrap();

    // sigmoid(0) is one half, so each argument's position shows in the result.
    assert_floats_close(&floats_of(&replayed), &[3.0, 6.0], FLOAT_TOLERANCE);
}

#[test]
fn should_compute_precise_swiglu_and_restore_the_input_element_type() {
    let context = gpu_context();
    let compiled = MlxCompiledElementwiseGraphs::new().expect("the graphs should compile");
    let up_states = context
        .astype(&float_array(&[3.0, 4.0], &[2]), MlxDtype::Float16)
        .unwrap();
    let gate_values = float_array(&[0.0, 2.0], &[2]);
    let gate_states = context.astype(&gate_values, MlxDtype::Float16).unwrap();

    let replayed = context
        .apply_compiled_precise_swiglu(&compiled, &up_states, &gate_states)
        .unwrap();
    let as_float32 = context.astype(&replayed, MlxDtype::Float32).unwrap();
    let expected_silu = floats_of(&context.silu(&gate_values).unwrap());

    assert_eq!(replayed.dtype(), MlxDtype::Float16);
    assert_floats_close(
        &floats_of(&as_float32),
        &[3.0 * expected_silu[0], 4.0 * expected_silu[1]],
        HALF_PRECISION_TOLERANCE,
    );
}

#[test]
fn should_cast_activations_to_float32_before_applying_signs() {
    let context = gpu_context();
    let compiled = MlxCompiledElementwiseGraphs::new().expect("the graphs should compile");
    let activations = context
        .astype(&float_array(&[1.5, 2.5], &[2]), MlxDtype::Float16)
        .unwrap();
    let signs = float_array(&[1.0, -1.0], &[2]);

    let replayed = context
        .apply_compiled_signed_float32_cast(&compiled, &activations, &signs)
        .unwrap();

    assert_eq!(replayed.dtype(), MlxDtype::Float32);
    assert_eq!(floats_of(&replayed), vec![1.5, -2.5]);
}

#[test]
fn should_rotate_the_head_halves_with_the_given_cosines_and_sines() {
    let context = gpu_context();
    let compiled = MlxCompiledElementwiseGraphs::new().expect("the graphs should compile");
    let attention_states = float_array(&[1.0, 2.0], &[1, 1, 2]);
    let cosines = float_array(&[0.5], &[1]);
    let sines = float_array(&[0.25], &[1]);
    let first_half = float_array(&[1.0], &[1, 1, 1]);
    let second_half = float_array(&[2.0], &[1, 1, 1]);

    let replayed = context
        .apply_compiled_vision_rope(
            &compiled,
            &attention_states,
            &cosines,
            &sines,
            &first_half,
            &second_half,
        )
        .unwrap();

    // states * cos + concat(-second, first) * sin = [0.5 - 0.5, 1.0 + 0.25].
    assert_floats_close(&floats_of(&replayed), &[0.0, 1.25], FLOAT_TOLERANCE);
}

#[test]
fn should_return_update_rates_first_and_matching_decay_factors_second() {
    let context = gpu_context();
    let single_outputs = MlxCompiledElementwiseGraphs::new().expect("the graphs should compile");
    let dual_output = MlxCompiledGatedDeltaRateActivation::new().expect("should compile");
    let update_logits = float_array(&[0.0], &[1]);
    let interval_inputs = float_array(&[1.0], &[1]);
    let rate_logarithm = float_array(&[0.5], &[1]);
    let interval_bias = float_array(&[2.0], &[1]);

    let (update_rates, decay_factors) = context
        .apply_compiled_gated_delta_rate_activation(
            &dual_output,
            &update_logits,
            &interval_inputs,
            &rate_logarithm,
            &interval_bias,
        )
        .unwrap();
    let single_decay = context
        .apply_compiled_gated_delta_decay(
            &single_outputs,
            &rate_logarithm,
            &interval_inputs,
            &interval_bias,
        )
        .unwrap();
    // The documented chain: exp(-exp(rate_log) * softplus(inputs + bias)).
    let softplus_of_interval = (1.0_f32 + 3.0_f32.exp()).ln();
    let expected_decay = (-0.5_f32.exp() * softplus_of_interval).exp();

    assert_eq!(floats_of(&update_rates), vec![0.5]);
    assert_floats_close(
        &floats_of(&decay_factors),
        &[expected_decay],
        FLOAT_TOLERANCE,
    );
    assert_eq!(floats_of(&decay_factors), floats_of(&single_decay));
}
