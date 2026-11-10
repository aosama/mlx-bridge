//! Tier 4 marshalling evidence for small linear algebra and FFT transforms.
//! MLX runs matrix factorization on the CPU only, so those contracts use the
//! CPU context and one contract proves the GPU context refuses them.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{
    FLOAT_TOLERANCE, array_at, assert_floats_close, cpu_context, float_array, floats_of,
    gpu_context,
};
use mlx_bridge::{MlxDtype, raw};

const BACKWARD_NORM: raw::mlx_fft_norm = raw::mlx_fft_norm__MLX_FFT_NORM_BACKWARD;
const ORTHO_NORM: raw::mlx_fft_norm = raw::mlx_fft_norm__MLX_FFT_NORM_ORTHO;

#[test]
fn should_invert_a_small_matrix_on_the_cpu_stream() {
    let context = cpu_context();
    let matrix = float_array(&[2.0, 0.0, 0.0, 4.0], &[2, 2]);

    let inverse = context.linalg_inv(&matrix).unwrap();

    assert_floats_close(
        &floats_of(&inverse),
        &[0.5, 0.0, 0.0, 0.25],
        FLOAT_TOLERANCE,
    );
}

#[test]
fn should_refuse_matrix_inversion_on_the_gpu_stream() {
    let context = gpu_context();
    let matrix = float_array(&[2.0, 0.0, 0.0, 4.0], &[2, 2]);

    let failure = context
        .linalg_inv(&matrix)
        .expect_err("inversion is CPU-only");

    assert!(failure.description.contains("GPU"), "{failure}");
}

#[test]
fn should_solve_a_small_system_in_argument_order() {
    let context = cpu_context();
    let coefficients = float_array(&[2.0, 0.0, 0.0, 4.0], &[2, 2]);
    let targets = float_array(&[2.0, 8.0], &[2]);

    let solution = context.linalg_solve(&coefficients, &targets).unwrap();

    assert_floats_close(&floats_of(&solution), &[1.0, 2.0], FLOAT_TOLERANCE);
}

#[test]
fn should_factor_through_the_requested_triangle() {
    let context = cpu_context();
    let matrix = float_array(&[4.0, 2.0, 2.0, 10.0], &[2, 2]);

    let lower = context.linalg_cholesky(&matrix, false).unwrap();
    let upper = context.linalg_cholesky(&matrix, true).unwrap();

    assert_floats_close(&floats_of(&lower), &[2.0, 0.0, 1.0, 3.0], FLOAT_TOLERANCE);
    assert_floats_close(&floats_of(&upper), &[2.0, 1.0, 0.0, 3.0], FLOAT_TOLERANCE);
}

#[test]
fn should_reconstruct_a_matrix_from_its_qr_factors() {
    let context = cpu_context();
    let matrix = float_array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let (orthogonal, triangular) = context.linalg_qr(&matrix).unwrap();
    let reconstructed = context.matmul(&orthogonal, &triangular).unwrap();

    assert_floats_close(&floats_of(&reconstructed), &floats_of(&matrix), 1e-4);
}

#[test]
fn should_return_only_singular_values_when_vectors_are_not_requested() {
    let context = cpu_context();
    let matrix = float_array(&[3.0, 0.0, 0.0, 2.0], &[2, 2]);

    let values_only = context.linalg_svd(&matrix, false).unwrap();
    let with_vectors = context.linalg_svd(&matrix, true).unwrap();

    assert_eq!(values_only.len(), 1);
    assert_floats_close(
        &floats_of(&array_at(&values_only, 0)),
        &[3.0, 2.0],
        FLOAT_TOLERANCE,
    );
    assert_eq!(with_vectors.len(), 3);
}

#[test]
fn should_read_symmetric_eigenvalues_through_the_requested_triangle() {
    let context = cpu_context();
    let matrix = float_array(&[2.0, 0.0, 0.0, 3.0], &[2, 2]);

    let eigenvalues = context.linalg_eigvalsh(&matrix, "L").unwrap();

    assert_floats_close(&floats_of(&eigenvalues), &[2.0, 3.0], FLOAT_TOLERANCE);
    assert!(context.linalg_eigvalsh(&matrix, "L\0").is_err());
}

#[test]
fn should_compute_the_requested_norm_on_any_stream() {
    let context = gpu_context();
    let vector = float_array(&[3.0, 4.0], &[2]);
    let matrix = float_array(&[1.0, 2.0, 2.0, 4.0], &[2, 2]);

    assert_eq!(
        floats_of(&context.linalg_norm_l2(&vector, &[], false).unwrap()),
        vec![5.0]
    );
    assert_eq!(
        floats_of(&context.linalg_norm(&vector, 1.0, &[], false).unwrap()),
        vec![7.0]
    );
    assert_eq!(
        floats_of(
            &context
                .linalg_norm_matrix(&matrix, "fro", &[], false)
                .unwrap()
        ),
        vec![5.0]
    );
}

#[test]
fn should_read_float64_values_built_on_the_cpu_stream_including_a_view() {
    let context = cpu_context();
    let single_precision = float_array(&[1.5, 2.5], &[2]);

    let double_precision = context
        .astype(&single_precision, MlxDtype::Float64)
        .unwrap();
    let reversed = context.flip(&double_precision).unwrap();

    assert_eq!(double_precision.to_vec_f64().unwrap(), vec![1.5, 2.5]);
    // The materializing copy must run on a stream that accepts float64.
    assert_eq!(reversed.to_vec_f64().unwrap(), vec![2.5, 1.5]);
}

#[test]
fn should_round_trip_a_signal_through_the_real_transforms() {
    let context = gpu_context();
    let signal = float_array(&[1.0, 2.0, 3.0, 4.0], &[4]);

    let spectrum = context.fft_rfft(&signal, 4, 0, BACKWARD_NORM).unwrap();
    let recovered = context.fft_irfft(&spectrum, 4, 0, BACKWARD_NORM).unwrap();

    assert_floats_close(&floats_of(&recovered), &[1.0, 2.0, 3.0, 4.0], 1e-4);
}

#[test]
fn should_apply_the_requested_normalization() {
    let context = gpu_context();
    let signal = float_array(&[1.0, 1.0, 1.0, 1.0], &[4]);

    let orthonormal = context.fft_rfft(&signal, 4, 0, ORTHO_NORM).unwrap();
    let real_parts = context.real(&orthonormal).unwrap();

    assert_floats_close(&floats_of(&real_parts)[..1], &[2.0], FLOAT_TOLERANCE);
}

#[test]
fn should_compute_frequencies_with_the_requested_spacing() {
    let context = gpu_context();

    let frequencies = context.fft_fftfreq(4, 0.5).unwrap();

    assert_floats_close(
        &floats_of(&frequencies),
        &[0.0, 0.5, -1.0, -0.5],
        FLOAT_TOLERANCE,
    );
}

#[test]
fn should_forward_the_length_argument_to_the_transform() {
    let context = gpu_context();
    let signal = float_array(&[1.0, 2.0, 3.0, 4.0], &[4]);

    let truncated = context.fft_rfft(&signal, 2, 0, BACKWARD_NORM).unwrap();

    assert_eq!(truncated.dtype(), MlxDtype::Complex64);
    assert_eq!(truncated.shape(), vec![2]);
}
