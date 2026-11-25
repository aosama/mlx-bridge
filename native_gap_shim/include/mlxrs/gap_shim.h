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
 * precedent). Where MLX-C has no type for a value, the shim declares one in
 * MLX-C's shape: a C enum for an MLX enum (mlxrs_dtype_kind) and a struct
 * holding a void* ctx for an owned C++ object (mlxrs_config_scope). The
 * prefix is mlxrs_ so shim symbols and types can never collide with a
 * future MLX-C one.
 *
 * Consumers who own their native image (default-features = false, or a
 * prefix of their own) compile the shim's sources in native_gap_shim/src
 * themselves: they are self-contained — public headers only, no MLX-C
 * private headers — and link the same mlx and mlxc libraries any consumer
 * already links.
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

/* Fused scaled dot-product attention with a sliding window (MLX
 * fast::scaled_dot_product_attention, the window_size parameter MLX v0.32.5
 * added). Every other argument means what it means in MLX-C's
 * mlx_fast_scaled_dot_product_attention. A query attends only to keys fewer
 * than `window_size` positions away; MLX rejects 0 and values below -1, and
 * -1 means no window. Retires when an MLX-C release forwards window_size;
 * the forward edit at MLX-C's own call site fails the build loudly then,
 * which is the signal to retire this entry too. */
int mlxrs_fast_scaled_dot_product_attention_windowed(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    int window_size,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_array sinks /* may be null */,
    bool force_fused,
    const mlx_stream s);

/* The element type two types promote to (MLX core promote_types). */
int mlxrs_promote_types(mlx_dtype* res, mlx_dtype first, mlx_dtype second);

/* MLX's coarse classification of an element type (MLX core Dtype::Kind). */
typedef enum mlxrs_dtype_kind_ {
  MLXRS_DTYPE_KIND_BOOL,
  MLXRS_DTYPE_KIND_UNSIGNED_INTEGER,
  MLXRS_DTYPE_KIND_SIGNED_INTEGER,
  MLXRS_DTYPE_KIND_FLOATING,
  MLXRS_DTYPE_KIND_COMPLEX,
  MLXRS_DTYPE_KIND_BRAIN_FLOATING,
} mlxrs_dtype_kind;

/* The element type's kind (MLX core kindof). */
int mlxrs_kindof(mlxrs_dtype_kind* res, mlx_dtype dtype);

/* MLX's element type hierarchy (MLX core Dtype::Category). */
typedef enum mlxrs_dtype_category_ {
  MLXRS_DTYPE_CATEGORY_COMPLEX_FLOATING,
  MLXRS_DTYPE_CATEGORY_FLOATING,
  MLXRS_DTYPE_CATEGORY_INEXACT,
  MLXRS_DTYPE_CATEGORY_SIGNED_INTEGER,
  MLXRS_DTYPE_CATEGORY_UNSIGNED_INTEGER,
  MLXRS_DTYPE_CATEGORY_INTEGER,
  MLXRS_DTYPE_CATEGORY_NUMBER,
  MLXRS_DTYPE_CATEGORY_GENERIC,
} mlxrs_dtype_category;

/* Whether the element type falls under the category (MLX core issubdtype,
 * the type-against-category overload). The other overloads are not carried:
 * a type is under another type only when the two are equal, and the
 * category hierarchy is fixed. */
int mlxrs_issubdtype(bool* res, mlx_dtype dtype, mlxrs_dtype_category category);

/* Whether MLX's Metal kernels can use the GPU neural accelerators (MLX
 * metal::is_nax_available). MLX answers false when it was built without the
 * Metal backend, so the query is safe on every host. */
int mlxrs_is_nax_available(bool* res);

/* Whether MLX's active memory is above its memory limit (MLX allocator
 * over_memory_limit). */
int mlxrs_over_memory_limit(bool* res);

/* Sets a configuration value for the whole process (MLX config::update).
 * The value outranks the environment variable of the same name. */
int mlxrs_config_update(const char* name, int value);

/* Reads a configuration value (MLX config::get): this thread's scoped
 * values first, then process values, then the environment variable of the
 * same name, then `default_value`. MLX records an environment or default
 * answer as the process value on the first read of a name. */
int mlxrs_config_get(int* res, const char* name, int default_value);

/* A thread-scoped set of configuration values (MLX config::ScopedUpdate). */
typedef struct mlxrs_config_scope_ {
  void* ctx;
} mlxrs_config_scope;

/* Applies `values_num` name and value pairs on the calling thread until the
 * scope is freed. MLX keeps scopes in a thread-local list, so the scope must
 * be freed on the thread that created it. */
int mlxrs_config_scope_new(
    mlxrs_config_scope* res,
    const char* const* names,
    const int* values,
    size_t values_num);

/* Removes the scope's values from the calling thread. */
int mlxrs_config_scope_free(mlxrs_config_scope scope);

#ifdef __cplusplus
}
#endif

#endif /* MLXRS_GAP_SHIM_H */
