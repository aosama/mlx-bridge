//! Gap-shim contracts: the repository's own mlxrs_* C ABI over the MLX
//! functions the vendored MLX-C pin has not wrapped, exercised end to end
//! through the bridge — the shim compiles in the native prefix, binds
//! through the combined parse, and behaves like any MLX-C family: real
//! values, captured errors, owned handles.

use mlx_c_rust::{
    MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxDtype, broadcast_shapes,
    einsum_path, install_non_terminating_error_handler, normalize_axis_index, raw,
};

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

const BACKWARD_NORM: raw::mlx_fft_norm = raw::mlx_fft_norm__MLX_FFT_NORM_BACKWARD;
const ORTHO_NORM: raw::mlx_fft_norm = raw::mlx_fft_norm__MLX_FFT_NORM_ORTHO;

fn eight_sample_ramp(context: &MlxBindingsContext) -> MlxArray {
    context
        .arange_f32(0.0, 8.0, 1.0)
        .expect("the ramp should build")
}

#[test]
fn should_forward_the_transform_length_and_hop_to_stft() {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new().expect("the bindings context should initialize");
    let signal = eight_sample_ramp(&context);

    let frames = context
        .stft(
            &signal,
            4,
            Some(4),
            None,
            None,
            false,
            "reflect",
            BACKWARD_NORM,
            None,
        )
        .expect("the graph should build");

    // MLX orders the axes (bins, frames): 4 / 2 + 1 one-sided bins, two frames.
    assert_eq!(frames.shape(), vec![3, 2]);
    assert_eq!(frames.dtype(), MlxDtype::Complex64);
}

#[test]
fn should_forward_the_onesided_choice_to_stft() {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new().expect("the bindings context should initialize");
    let signal = eight_sample_ramp(&context);

    let two_sided = context
        .stft(
            &signal,
            4,
            Some(4),
            None,
            None,
            false,
            "reflect",
            BACKWARD_NORM,
            Some(false),
        )
        .expect("the graph should build");

    assert_eq!(two_sided.shape(), vec![4, 2]);
}

#[test]
fn should_forward_the_centering_choice_and_window_to_stft() {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new().expect("the bindings context should initialize");
    let signal = eight_sample_ramp(&context);
    let zero_window = context.zeros(&[4], MlxDtype::Float32).unwrap();

    let centered = context
        .stft(
            &signal,
            4,
            Some(4),
            None,
            None,
            true,
            "reflect",
            BACKWARD_NORM,
            None,
        )
        .expect("the graph should build");
    let windowed_out = context
        .stft(
            &signal,
            4,
            Some(4),
            None,
            Some(&zero_window),
            false,
            "reflect",
            BACKWARD_NORM,
            None,
        )
        .expect("the graph should build");

    assert_eq!(centered.shape(), vec![3, 3]);
    let real_parts = context.real(&windowed_out).unwrap().to_vec_f32().unwrap();
    for real_part in real_parts {
        assert_eq!(real_part, 0.0, "a zero window must silence every frame");
    }
}

#[test]
fn should_round_trip_a_signal_through_stft_and_istft() {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new().expect("the bindings context should initialize");
    let signal = eight_sample_ramp(&context);

    let frames = context
        .stft(
            &signal,
            4,
            Some(2),
            None,
            None,
            true,
            "reflect",
            ORTHO_NORM,
            None,
        )
        .expect("the graph should build");
    let recovered = context
        .istft(
            &frames,
            Some(4),
            Some(2),
            None,
            None,
            true,
            ORTHO_NORM,
            None,
            Some(8),
        )
        .expect("the graph should build");

    assert_eq!(recovered.shape(), vec![8]);
    let recovered_values = recovered.to_vec_f32().unwrap();
    let original_values = signal.to_vec_f32().unwrap();
    for (recovered_value, original_value) in recovered_values.iter().zip(&original_values) {
        assert!(
            (recovered_value - original_value).abs() < 1e-3,
            "{recovered_values:?}"
        );
    }
}

#[test]
fn should_reject_a_pad_mode_with_an_interior_null_byte() {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new().expect("the bindings context should initialize");
    let signal = eight_sample_ramp(&context);

    let failure = context
        .stft(
            &signal,
            4,
            None,
            None,
            None,
            true,
            "a\0b",
            BACKWARD_NORM,
            None,
        )
        .expect_err("a C string cannot hold a null byte");

    assert_eq!(failure.operation, "apply the gap shim stft");
}

#[test]
fn should_broadcast_two_shapes_and_refuse_incompatible_ones() {
    install_non_terminating_error_handler();

    assert_eq!(broadcast_shapes(&[2, 1], &[3]).unwrap(), vec![2, 3]);
    assert_eq!(broadcast_shapes(&[], &[4]).unwrap(), vec![4]);
    assert_eq!(broadcast_shapes(&[], &[]).unwrap(), Vec::<i32>::new());
    assert!(broadcast_shapes(&[2], &[3]).is_err());
}

#[test]
fn should_report_the_contraction_path_einsum_would_take() {
    install_non_terminating_error_handler();
    let left = MlxArray::from_f32(&[1.0; 4], &[2, 2]).unwrap();
    let right = MlxArray::from_f32(&[1.0; 4], &[2, 2]).unwrap();
    let operands = MlxArrayVector::new(&[&left, &right]).unwrap();

    let path = einsum_path("ij,jk->ik", &operands).unwrap();

    assert_eq!(path.steps, vec![vec![0, 1]]);
    assert!(
        path.description.contains("ij,jk->ik"),
        "{}",
        path.description
    );
}

#[test]
fn should_refuse_invalid_einsum_subscripts_when_planning_a_path() {
    install_non_terminating_error_handler();
    let left = MlxArray::from_f32(&[1.0], &[1]).unwrap();
    let operands = MlxArrayVector::new(&[&left]).unwrap();

    assert!(einsum_path("not valid", &operands).is_err());
    assert!(einsum_path("i\0i", &operands).is_err());
}

#[test]
fn should_normalize_axes_from_either_end_and_refuse_out_of_range_ones() {
    install_non_terminating_error_handler();

    assert_eq!(normalize_axis_index(1, 3).unwrap(), 1);
    assert_eq!(normalize_axis_index(-1, 3).unwrap(), 2);
    assert!(normalize_axis_index(3, 3).is_err());
    assert!(normalize_axis_index(-4, 3).is_err());
}
