/* The mlx-bridge gap shim's array-level entries: operations, transforms,
 * FFT, element-type queries, and formatting. Process-level queries live in
 * gap_shim_runtime.cpp. See gap_shim.h for the contract and the retirement
 * policy.
 */

#include "mlxrs/gap_shim.h"

#include "gap_shim_handles.h"
#include "mlx/c/error.h"
#include "mlx/ops.h"

using namespace mlxrs_shim;

extern "C" int mlxrs_minmax_axes(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) {
  try {
    auto [minima, maxima] = mlx::core::minmax(
        shim_array_get(a),
        std::vector<int>(axes, axes + axes_num),
        keepdims,
        shim_stream_get(s));
    shim_array_set(*res_0, std::move(minima));
    shim_array_set(*res_1, std::move(maxima));
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_minmax_axis(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) {
  try {
    auto [minima, maxima] =
        mlx::core::minmax(shim_array_get(a), axis, keepdims, shim_stream_get(s));
    shim_array_set(*res_0, std::move(minima));
    shim_array_set(*res_1, std::move(maxima));
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_vmap(
    mlx_closure* res,
    const mlx_closure fun,
    const int* in_axes,
    size_t in_axes_num,
    const int* out_axes,
    size_t out_axes_num) {
  try {
    if (!fun.ctx) {
      throw std::runtime_error("expected a non-empty mlx_closure");
    }
    ShimClosureFunction vectorized = mlx::core::vmap(
        *static_cast<ShimClosureFunction*>(fun.ctx),
        std::vector<int>(in_axes, in_axes + in_axes_num),
        std::vector<int>(out_axes, out_axes + out_axes_num));
    // MLX-C frees a closure handle as a pointer to this exact function type.
    if (res->ctx) {
      *static_cast<ShimClosureFunction*>(res->ctx) = std::move(vectorized);
    } else {
      res->ctx = new ShimClosureFunction(std::move(vectorized));
    }
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_stft(
    mlx_array* res,
    const mlx_array x,
    int n_fft,
    mlx_optional_int hop_length,
    mlx_optional_int win_length,
    const mlx_array window,
    bool center,
    const char* pad_mode,
    mlx_fft_norm norm,
    bool has_onesided,
    bool onesided,
    const mlx_stream s) {
  try {
    shim_array_set(
        *res,
        mlx::core::fft::stft(
            shim_array_get(x),
            n_fft,
            shim_optional_int(hop_length),
            shim_optional_int(win_length),
            shim_optional_array(window),
            center,
            std::string(pad_mode),
            shim_fft_norm(norm),
            has_onesided ? std::make_optional(onesided) : std::nullopt,
            shim_stream_get(s)));
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_istft(
    mlx_array* res,
    const mlx_array stft_matrix,
    mlx_optional_int n_fft,
    mlx_optional_int hop_length,
    mlx_optional_int win_length,
    const mlx_array window,
    bool center,
    mlx_fft_norm norm,
    bool has_onesided,
    bool onesided,
    mlx_optional_int length,
    const mlx_stream s) {
  try {
    shim_array_set(
        *res,
        mlx::core::fft::istft(
            shim_array_get(stft_matrix),
            shim_optional_int(n_fft),
            shim_optional_int(hop_length),
            shim_optional_int(win_length),
            shim_optional_array(window),
            center,
            shim_fft_norm(norm),
            has_onesided ? std::make_optional(onesided) : std::nullopt,
            shim_optional_int(length),
            shim_stream_get(s)));
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_result_type(mlx_dtype* res, const mlx_vector_array arrays) {
  try {
    *res = shim_dtype_to_c(mlx::core::result_type(shim_vector_array_get(arrays)));
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_broadcast_shapes(
    mlx_vector_int* res,
    const int* first_shape,
    size_t first_shape_num,
    const int* second_shape,
    size_t second_shape_num) {
  try {
    const mlx::core::Shape broadcast = mlx::core::broadcast_shapes(
        mlx::core::Shape(first_shape, first_shape + first_shape_num),
        mlx::core::Shape(second_shape, second_shape + second_shape_num));
    shim_vector_int_set(res, std::vector<int>(broadcast.begin(), broadcast.end()));
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_einsum_path(
    mlx_vector_int* flattened_path,
    mlx_vector_int* path_step_lengths,
    mlx_string* description,
    const char* subscripts,
    const mlx_vector_array operands) {
  try {
    auto [path, path_description] = mlx::core::einsum_path(
        std::string(subscripts), shim_vector_array_get(operands));
    std::vector<int> flattened;
    std::vector<int> step_lengths;
    for (const std::vector<int>& step : path) {
      flattened.insert(flattened.end(), step.begin(), step.end());
      step_lengths.push_back(static_cast<int>(step.size()));
    }
    shim_vector_int_set(flattened_path, flattened);
    shim_vector_int_set(path_step_lengths, step_lengths);
    shim_string_set(description, path_description);
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_normalize_axis_index(int* res, int axis, int ndim) {
  try {
    *res = mlx::core::normalize_axis_index(axis, ndim);
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_peek_default_stream(
    mlx_stream* res,
    bool* has_stream,
    const mlx_device device) {
  try {
    if (!device.ctx) {
      throw std::runtime_error("expected a non-empty mlx_device");
    }
    const std::optional<mlx::core::Stream> stream = mlx::core::peek_default_stream(
        *static_cast<mlx::core::Device*>(device.ctx));
    *has_stream = stream.has_value();
    if (stream.has_value()) {
      res->ctx = new mlx::core::Stream(*stream);
    }
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_set_printoptions(int precision, bool capitalize_bool) {
  try {
    mlx::core::set_printoptions(mlx::core::PrintOptions{precision});
    mlx::core::get_global_formatter().capitalize_bool = capitalize_bool;
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_get_global_formatter(int* precision, bool* capitalize_bool) {
  try {
    const mlx::core::PrintFormatter& formatter = mlx::core::get_global_formatter();
    *precision = formatter.format_options.precision;
    *capitalize_bool = formatter.capitalize_bool;
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_finfo(
    int* bits,
    double* min,
    double* max,
    double* eps,
    double* smallest_normal,
    mlx_dtype dtype) {
  try {
    const mlx::core::finfo limits(shim_dtype_from_c(dtype));
    *bits = limits.bits;
    *min = limits.min;
    *max = limits.max;
    *eps = limits.eps;
    *smallest_normal = limits.smallest_normal;
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_fast_scaled_dot_product_attention_windowed(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    int window_size,
    const char* mask_mode,
    const mlx_array mask_arr,
    const mlx_array sinks,
    bool force_fused,
    const mlx_stream s) {
  try {
    shim_array_set(
        *res,
        mlx::core::fast::scaled_dot_product_attention(
            shim_array_get(queries),
            shim_array_get(keys),
            shim_array_get(values),
            scale,
            window_size,
            std::string(mask_mode),
            shim_optional_array(mask_arr),
            shim_optional_array(sinks),
            force_fused,
            shim_stream_get(s)));
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_promote_types(mlx_dtype* res, mlx_dtype first, mlx_dtype second) {
  try {
    *res = shim_dtype_to_c(mlx::core::promote_types(
        shim_dtype_from_c(first), shim_dtype_from_c(second)));
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_kindof(mlxrs_dtype_kind* res, mlx_dtype dtype) {
  try {
    *res = shim_dtype_kind_to_c(mlx::core::kindof(shim_dtype_from_c(dtype)));
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_issubdtype(
    bool* res,
    mlx_dtype dtype,
    mlxrs_dtype_category category) {
  try {
    *res = mlx::core::issubdtype(
        shim_dtype_from_c(dtype), shim_dtype_category_from_c(category));
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}
