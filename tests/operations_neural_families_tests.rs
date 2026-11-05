//! Tier 4 marshalling evidence for the model-facing families: small fixtures
//! whose answers change if an argument lands in the wrong C parameter. The
//! families' numerics belong to MLX's own suite.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{
    FLOAT_TOLERANCE, array_at, assert_floats_close, float_array, floats_of, gpu_context, int_array,
};
use mlx_c_rust::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxDtype};

fn ramp_weights(context: &MlxBindingsContext) -> MlxArray {
    let ramp = context.arange_f32(0.0, 256.0, 1.0).unwrap();
    let scaled = context.multiply_scalar(&ramp, 0.01).unwrap();
    context.reshape(&scaled, &[4, 64]).unwrap()
}

#[test]
fn should_round_trip_weights_through_affine_quantization_within_one_step() {
    let context = gpu_context();
    let weights = ramp_weights(&context);

    let (quantized, scales, biases) = context.quantize_affine(&weights, 32, 8).unwrap();
    let restored = context
        .dequantize_affine(&quantized, &scales, &biases, 32, 8)
        .unwrap();

    assert_eq!(restored.shape(), weights.shape());
    // Each group spans 0.31 across 255 levels, so one level is about 0.0012.
    assert_floats_close(&floats_of(&restored), &floats_of(&weights), 0.002);
}

#[test]
fn should_start_rotating_from_the_requested_token_offset() {
    let context = gpu_context();
    let token = float_array(&[1.0, 0.0], &[1, 1, 2]);
    let two_tokens = float_array(&[1.0, 0.0, 1.0, 0.0], &[1, 2, 2]);

    let offset_rotation = floats_of(&context.rope(&token, 2, 10_000.0, 1).unwrap());
    let second_token_rotation = floats_of(&context.rope(&two_tokens, 2, 10_000.0, 0).unwrap());

    assert_floats_close(
        &offset_rotation,
        &second_token_rotation[2..],
        FLOAT_TOLERANCE,
    );
}

#[test]
fn should_return_the_only_value_when_attention_has_one_key() {
    let context = gpu_context();
    let query = float_array(&[1.0, 0.0], &[1, 1, 1, 2]);
    let key = float_array(&[0.0, 1.0], &[1, 1, 1, 2]);
    let value = float_array(&[3.0, 4.0], &[1, 1, 1, 2]);

    let attended = context
        .scaled_dot_product_attention(&query, &key, &value, 1.0)
        .unwrap();

    assert_floats_close(&floats_of(&attended), &[3.0, 4.0], FLOAT_TOLERANCE);
}

#[test]
fn should_exclude_masked_keys_from_attention() {
    let context = gpu_context();
    let query = float_array(&[1.0, 0.0], &[1, 1, 1, 2]);
    let keys = float_array(&[1.0, 0.0, 1.0, 0.0], &[1, 1, 2, 2]);
    let values = float_array(&[2.0, 2.0, 4.0, 6.0], &[1, 1, 2, 2]);
    let mask_scores = float_array(&[0.0, 1.0], &[1, 1, 1, 2]);
    let keep_second_only = context
        .greater(&mask_scores, &MlxArray::from_scalar_f32(0.5))
        .unwrap();

    let attended = context
        .masked_scaled_dot_product_attention(&query, &keys, &values, 1.0, &keep_second_only)
        .unwrap();

    assert_floats_close(&floats_of(&attended), &[4.0, 6.0], FLOAT_TOLERANCE);
}

#[test]
fn should_hide_later_keys_from_earlier_queries_through_the_causal_entry() {
    let context = gpu_context();
    let queries = float_array(&[1.0, 0.0, 1.0, 0.0], &[1, 1, 2, 2]);
    let values = float_array(&[2.0, 2.0, 4.0, 6.0], &[1, 1, 2, 2]);

    let attended = context
        .causal_scaled_dot_product_attention(&queries, &queries, &values, 1.0)
        .unwrap();

    assert_floats_close(&floats_of(&attended)[..2], &[2.0, 2.0], FLOAT_TOLERANCE);
}

#[test]
fn should_slide_a_convolution_kernel_with_the_requested_stride() {
    let context = gpu_context();
    let signal = float_array(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1]);
    let adjacent_sum_kernel = float_array(&[1.0, 1.0], &[1, 2, 1]);

    let unit_stride = context
        .conv1d(&signal, &adjacent_sum_kernel, 1, 0, 1, 1)
        .unwrap();
    let double_stride = context
        .conv1d(&signal, &adjacent_sum_kernel, 2, 0, 1, 1)
        .unwrap();

    assert_eq!(unit_stride.shape(), vec![1, 3, 1]);
    assert_eq!(floats_of(&unit_stride), vec![3.0, 5.0, 7.0]);
    assert_eq!(floats_of(&double_stride), vec![3.0, 7.0]);
}

#[test]
fn should_apply_the_requested_padding_to_a_convolution() {
    let context = gpu_context();
    let signal = float_array(&[1.0, 2.0], &[1, 2, 1]);
    let adjacent_sum_kernel = float_array(&[1.0, 1.0], &[1, 2, 1]);

    let padded = context
        .conv1d(&signal, &adjacent_sum_kernel, 1, 1, 1, 1)
        .unwrap();

    assert_eq!(floats_of(&padded), vec![1.0, 3.0, 2.0]);
}

#[test]
fn should_compute_the_activations_at_their_known_points() {
    let context = gpu_context();
    let zero = float_array(&[0.0], &[1]);

    assert_eq!(floats_of(&context.sigmoid(&zero).unwrap()), vec![0.5]);
    assert_eq!(floats_of(&context.silu(&zero).unwrap()), vec![0.0]);
    assert_floats_close(
        &floats_of(&context.softplus(&zero).unwrap()),
        &[std::f32::consts::LN_2],
        FLOAT_TOLERANCE,
    );
}

#[test]
fn should_normalize_along_the_final_axis_with_a_unit_weight() {
    let context = gpu_context();
    let input = float_array(&[3.0, 3.0], &[1, 2]);
    let unit_weight = float_array(&[1.0, 1.0], &[2]);

    let normalized = context.rms_norm(&input, &unit_weight, 0.0).unwrap();

    assert_floats_close(&floats_of(&normalized), &[1.0, 1.0], FLOAT_TOLERANCE);
}

#[test]
fn should_gather_and_multiply_at_the_requested_indices() {
    let context = gpu_context();
    let left = context.zeros(&[2, 1, 2], MlxDtype::Float32).unwrap();
    let right = context.zeros(&[2, 2, 1], MlxDtype::Float32).unwrap();
    let left_indices = int_array(&[1, 0, 1], &[3]);
    let right_indices = int_array(&[0, 0, 1], &[3]);

    let product = context
        .gather_dense_matmul(
            &left,
            &right,
            Some(&left_indices),
            Some(&right_indices),
            false,
        )
        .unwrap();

    assert_eq!(product.shape(), vec![3, 1, 1]);
}

#[test]
fn should_forward_the_group_size_and_bit_width_to_quantization() {
    let context = gpu_context();
    let weights = ramp_weights(&context);

    let (eight_bit_weights, eight_bit_scales, _biases) =
        context.quantize_affine(&weights, 32, 8).unwrap();
    let (four_bit_weights, _scales, _biases) = context.quantize_affine(&weights, 32, 4).unwrap();
    let (_weights, wider_group_scales, _biases) = context.quantize_affine(&weights, 64, 8).unwrap();

    // MLX packs 32 / bits elements per word and keeps one scale per group, so
    // these shapes show the arguments arrived without restating its numerics.
    assert_eq!(eight_bit_weights.shape(), vec![4, 16]);
    assert_eq!(four_bit_weights.shape(), vec![4, 8]);
    assert_eq!(eight_bit_scales.shape(), vec![4, 2]);
    assert_eq!(wider_group_scales.shape(), vec![4, 1]);
}

/// The argument shapes all differ, so a swapped parameter fails to build.
struct GatedDeltaFixture {
    queries: MlxArray,
    keys: MlxArray,
    values: MlxArray,
    gates: MlxArray,
    beta: MlxArray,
}

impl GatedDeltaFixture {
    fn with_distinct_shapes(context: &MlxBindingsContext) -> Self {
        Self {
            queries: context.full(&[1, 3, 1, 4], 0.5, MlxDtype::Float32).unwrap(),
            keys: context.full(&[1, 3, 1, 4], 0.5, MlxDtype::Float32).unwrap(),
            values: context.full(&[1, 3, 2, 5], 1.0, MlxDtype::Float32).unwrap(),
            gates: context.full(&[1, 3, 2], 0.5, MlxDtype::Float32).unwrap(),
            beta: context.full(&[1, 3, 2], 0.5, MlxDtype::Float32).unwrap(),
        }
    }

    fn update(
        &self,
        context: &MlxBindingsContext,
        initial_state: Option<&MlxArray>,
        mask: Option<&MlxArray>,
    ) -> MlxArrayVector {
        context
            .gated_delta_update(
                &self.queries,
                &self.keys,
                &self.values,
                &self.gates,
                &self.beta,
                initial_state,
                mask,
            )
            .expect("the gated-delta graph should build")
    }
}

#[test]
fn should_return_gated_delta_outputs_before_the_recurrent_state() {
    let context = gpu_context();
    let fixture = GatedDeltaFixture::with_distinct_shapes(&context);

    let outputs = fixture.update(&context, None, None);

    assert_eq!(outputs.len(), 2);
    assert_eq!(array_at(&outputs, 0).shape(), vec![1, 3, 2, 5]);
    assert_eq!(array_at(&outputs, 1).shape(), vec![1, 2, 5, 4]);
}

#[test]
fn should_forward_the_initial_state_to_the_gated_delta_update() {
    let context = gpu_context();
    let fixture = GatedDeltaFixture::with_distinct_shapes(&context);
    let zero_state = context.zeros(&[1, 2, 5, 4], MlxDtype::Float32).unwrap();
    let nonzero_state = context.full(&[1, 2, 5, 4], 1.0, MlxDtype::Float32).unwrap();

    let without_state = fixture.update(&context, None, None);
    let from_zero = fixture.update(&context, Some(&zero_state), None);
    let from_nonzero = fixture.update(&context, Some(&nonzero_state), None);

    let without_state_outputs = floats_of(&array_at(&without_state, 0));
    assert_floats_close(
        &without_state_outputs,
        &floats_of(&array_at(&from_zero, 0)),
        FLOAT_TOLERANCE,
    );
    assert_ne!(
        without_state_outputs,
        floats_of(&array_at(&from_nonzero, 0))
    );
}

#[test]
fn should_forward_the_mask_to_the_gated_delta_update() {
    let context = gpu_context();
    let fixture = GatedDeltaFixture::with_distinct_shapes(&context);
    let nothing_selected = context
        .greater(
            &context.zeros(&[1, 3], MlxDtype::Float32).unwrap(),
            &MlxArray::from_scalar_f32(0.5),
        )
        .unwrap();

    let masked = fixture.update(&context, None, Some(&nothing_selected));

    // With no step selected the state never leaves its zero start.
    for state_value in floats_of(&array_at(&masked, 1)) {
        assert_eq!(state_value, 0.0);
    }
}

#[test]
fn should_translate_swapped_gated_delta_arguments_into_an_error() {
    let context = gpu_context();
    let fixture = GatedDeltaFixture::with_distinct_shapes(&context);

    let failure = context
        .gated_delta_update(
            &fixture.values,
            &fixture.keys,
            &fixture.queries,
            &fixture.gates,
            &fixture.beta,
            None,
            None,
        )
        .expect_err("the shapes only line up in the documented order");

    assert_eq!(failure.operation, "apply the MLX gated-delta update");
}

#[test]
fn should_compute_cross_entropy_from_logits_and_targets_in_order() {
    let context = gpu_context();
    let uniform_logits = float_array(&[0.0, 0.0], &[1, 2]);
    let target = int_array(&[1], &[1]);

    let loss = context
        .fast_cross_entropy(&uniform_logits, &target)
        .unwrap();

    // Uniform logits over two classes leave ln 2 whichever class is the target.
    assert_floats_close(
        &floats_of(&loss),
        &[std::f32::consts::LN_2],
        FLOAT_TOLERANCE,
    );
    assert!(
        context
            .fast_cross_entropy(&target, &uniform_logits)
            .is_err()
    );
}
