//! Tier 5 contracts for platform queries: availability that agrees with an
//! actual computation. Distributed group operations and CUDA runtime paths are
//! excluded by issue scope: they need cooperating processes and hardware these
//! runners cannot provide.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{
    float_array, floats_of, gpu_context, linked_mlx_has_a_gpu_backend, skip_without_gpu_backend,
};
use mlx_bridge::{
    clear_distributed_backends, cuda_is_available, distributed_is_available,
    initialize_distributed, metal_is_available, metal_is_nax_available, set_wired_limit,
};

#[test]
fn should_report_no_metal_on_a_build_without_a_gpu_backend() {
    // The other half of the agreement below: where the suites run on the CPU
    // because MLX has no GPU backend, MLX must not claim Metal either.
    if linked_mlx_has_a_gpu_backend() {
        return;
    }
    gpu_context();

    assert!(!metal_is_available().unwrap());
}

#[test]
fn should_agree_on_metal_availability_with_an_actual_gpu_computation() {
    if skip_without_gpu_backend("should_agree_on_metal_availability_with_an_actual_gpu_computation")
    {
        return;
    }
    let context = gpu_context();
    let values = float_array(&[1.0, 2.0], &[2]);

    let doubled = context.add(&values, &values).unwrap();

    assert!(metal_is_available().unwrap());
    assert_eq!(floats_of(&doubled), vec![2.0, 4.0]);
}

#[test]
fn should_report_no_neural_accelerators_on_a_build_without_a_gpu_backend() {
    // Negative: MLX's no-Metal build answers false, and the wrapper must
    // carry that answer instead of an error or a stale output value.
    if linked_mlx_has_a_gpu_backend() {
        return;
    }
    gpu_context();

    assert!(!metal_is_nax_available().unwrap());
}

#[test]
fn should_answer_the_neural_accelerator_query_the_same_way_every_time() {
    // The answer depends on the machine's GPU generation and operating
    // system, so no value can be asserted here. What the wrapper owes is a
    // successful status and the value MLX wrote, which MLX computes once per
    // process.
    if skip_without_gpu_backend(
        "should_answer_the_neural_accelerator_query_the_same_way_every_time",
    ) {
        return;
    }
    gpu_context();

    let first_answer = metal_is_nax_available().unwrap();
    let second_answer = metal_is_nax_available().unwrap();

    assert_eq!(first_answer, second_answer);
}

#[test]
fn should_never_report_neural_accelerators_without_metal() {
    // Boundary between the two queries: accelerated kernels are Metal
    // kernels, so the accelerator answer can never exceed Metal's.
    gpu_context();

    let metal_available = metal_is_available().unwrap();
    let neural_accelerators_available = metal_is_nax_available().unwrap();

    assert!(metal_available || !neural_accelerators_available);
}

#[test]
fn should_report_no_cuda_support_on_apple_silicon() {
    gpu_context();

    assert!(!cuda_is_available().unwrap());
}

#[test]
fn should_answer_a_distributed_availability_query_and_reject_a_null_byte() {
    gpu_context();

    assert!(distributed_is_available("").is_ok());
    assert!(distributed_is_available("a\0b").is_err());
}

#[test]
fn should_return_the_previous_wired_limit_when_it_changes() {
    if skip_without_gpu_backend("should_return_the_previous_wired_limit_when_it_changes") {
        return;
    }
    gpu_context();
    const FIRST_WIRED_LIMIT_BYTES: usize = 1024 * 1024;
    let original_limit = set_wired_limit(FIRST_WIRED_LIMIT_BYTES).unwrap();

    let previous_limit = set_wired_limit(original_limit).unwrap();

    assert_eq!(previous_limit, FIRST_WIRED_LIMIT_BYTES);
}

#[test]
fn should_report_this_process_as_the_only_member_of_a_singleton_group() {
    gpu_context();

    let group = initialize_distributed(false, "any").unwrap();

    assert_eq!(group.rank().unwrap(), 0);
    assert_eq!(group.size().unwrap(), 1);
}

#[test]
fn should_reinitialize_after_clearing_the_distributed_backends() {
    gpu_context();
    drop(initialize_distributed(false, "any").unwrap());

    clear_distributed_backends().unwrap();

    assert_eq!(
        initialize_distributed(false, "any")
            .unwrap()
            .size()
            .unwrap(),
        1
    );
}

#[test]
fn should_refuse_an_unknown_or_malformed_backend_name() {
    gpu_context();

    assert!(initialize_distributed(false, "not-a-backend").is_err());
    assert!(initialize_distributed(false, "a\0b").is_err());
}
