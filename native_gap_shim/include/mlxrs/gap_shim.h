#ifndef MLXRS_GAP_SHIM_H
#define MLXRS_GAP_SHIM_H

/* The mlx-bridge gap shim: this repository's own C ABI over the pinned
 * MLX C++ surface, carrying exactly the functions the vendored MLX-C pin
 * has not wrapped yet. It exists so an MLX capability becomes reachable
 * from Rust the day the MLX pin moves, instead of the day MLX-C happens
 * to catch up; when an MLX-C release wraps a function, the shim entry
 * retires (the audit-upstream-surface tool flags redundant entries).
 *
 * Conventions are MLX-C's own, so the bridge treats shim entry points
 * exactly like MLX-C ones: status-returning functions that report through
 * MLX's installed error handler, MLX-C handle types in and out, and
 * multi-output results as res_0/res_1 out parameters (the eig/qr
 * precedent). The prefix is mlxrs_ so shim symbols can never collide with
 * a future MLX-C symbol.
 *
 * Consumers who own their native image (default-features = false, or a
 * prefix of their own) compile this one-file shim themselves: it is
 * self-contained — public headers only, no MLX-C private headers — and
 * links the same mlx and mlxc libraries any consumer already links.
 */

#include <stdbool.h>
#include <stddef.h>

#include "mlx/c/mlx.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The element-wise minimum and maximum of `a` reduced over `axes`
 * (MLX core minmax, the axes overload). res_0 receives the minima,
 * res_1 the maxima. */
int mlxrs_minmax_axes(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s);

/* The minimum and maximum of `a` along one `axis` (MLX core minmax, the
 * axis overload). res_0 receives the minima, res_1 the maxima. */
int mlxrs_minmax_axis(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s);

/* Vectorizes a closure over the requested axes (MLX core vmap). An empty
 * axes list means axis 0 for every argument or output. */
int mlxrs_vmap(
    mlx_closure* res,
    const mlx_closure fun,
    const int* in_axes,
    size_t in_axes_num,
    const int* out_axes,
    size_t out_axes_num);

/* The short-time Fourier transform of the last axis of `x` (MLX fft::stft).
 * `has_onesided` selects between `onesided` and MLX's default. */
int mlxrs_stft(
    mlx_array* res,
    const mlx_array x,
    int n_fft,
    mlx_optional_int hop_length,
    mlx_optional_int win_length,
    const mlx_array window /* may be null */,
    bool center,
    const char* pad_mode,
    mlx_fft_norm norm,
    bool has_onesided,
    bool onesided,
    const mlx_stream s);

/* The inverse short-time Fourier transform (MLX fft::istft). */
int mlxrs_istft(
    mlx_array* res,
    const mlx_array stft_matrix,
    mlx_optional_int n_fft,
    mlx_optional_int hop_length,
    mlx_optional_int win_length,
    const mlx_array window /* may be null */,
    bool center,
    mlx_fft_norm norm,
    bool has_onesided,
    bool onesided,
    mlx_optional_int length,
    const mlx_stream s);

/* The element type the arrays promote to (MLX core result_type). */
int mlxrs_result_type(mlx_dtype* res, const mlx_vector_array arrays);

/* The shape two shapes broadcast to (MLX core broadcast_shapes). */
int mlxrs_broadcast_shapes(
    mlx_vector_int* res,
    const int* first_shape,
    size_t first_shape_num,
    const int* second_shape,
    size_t second_shape_num);

/* The contraction path MLX would take (MLX core einsum_path). C has no nested
 * integer vector, so the path arrives flattened alongside each step's length. */
int mlxrs_einsum_path(
    mlx_vector_int* flattened_path,
    mlx_vector_int* path_step_lengths,
    mlx_string* description,
    const char* subscripts,
    const mlx_vector_array operands);

/* The axis normalized into [0, ndim) (MLX core normalize_axis_index). */
int mlxrs_normalize_axis_index(int* res, int axis, int ndim);

/* The device's default stream on this thread, if one exists yet, without
 * creating it (MLX core peek_default_stream). `res` is written only when
 * `has_stream` comes back true. */
int mlxrs_peek_default_stream(
    mlx_stream* res,
    bool* has_stream,
    const mlx_device device);

/* Sets how arrays print (MLX core set_printoptions, plus the formatter's
 * boolean capitalization). A negative precision selects MLX's default. */
int mlxrs_set_printoptions(int precision, bool capitalize_bool);

/* Reads how arrays print (MLX core get_global_formatter). */
int mlxrs_get_global_formatter(int* precision, bool* capitalize_bool);

/* A floating-point element type's limits (MLX core finfo). */
int mlxrs_finfo(
    int* bits,
    double* min,
    double* max,
    double* eps,
    double* smallest_normal,
    mlx_dtype dtype);

#ifdef __cplusplus
}
#endif

#endif /* MLXRS_GAP_SHIM_H */
