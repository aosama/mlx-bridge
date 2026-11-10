/* The mlx-bridge gap shim implementation. See gap_shim.h for the contract
 * and the retirement policy. The handle casts below inline what MLX-C's
 * private headers do: the public C struct carries the C++ object as a
 * void* ctx (that layout is the published C ABI), so casting ctx is both
 * self-contained — this file compiles against any installed MLX-C, no
 * private headers — and exactly what the vendored MLX-C itself does.
 */

#include "mlxrs/gap_shim.h"

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mlx/c/error.h"
#include "mlx/mlx.h"
#include "mlx/ops.h"

namespace {

inline mlx::core::array& shim_array_get(const mlx_array handle) {
  if (!handle.ctx) {
    throw std::runtime_error("expected a non-empty mlx_array");
  }
  return *static_cast<mlx::core::array*>(handle.ctx);
}

inline void shim_array_set(mlx_array& destination, mlx::core::array&& value) {
  if (destination.ctx) {
    *static_cast<mlx::core::array*>(destination.ctx) = std::move(value);
  } else {
    destination.ctx = new mlx::core::array(std::move(value));
  }
}

inline mlx::core::Stream& shim_stream_get(const mlx_stream handle) {
  if (!handle.ctx) {
    throw std::runtime_error("expected a non-empty mlx_stream");
  }
  return *static_cast<mlx::core::Stream*>(handle.ctx);
}

using ShimClosureFunction = std::function<std::vector<mlx::core::array>(
    const std::vector<mlx::core::array>&)>;

inline std::vector<mlx::core::array>& shim_vector_array_get(
    const mlx_vector_array handle) {
  if (!handle.ctx) {
    throw std::runtime_error("expected a non-empty mlx_vector_array");
  }
  return *static_cast<std::vector<mlx::core::array>*>(handle.ctx);
}

inline std::optional<int> shim_optional_int(const mlx_optional_int value) {
  return value.has_value ? std::make_optional(value.value) : std::nullopt;
}

inline std::optional<mlx::core::array> shim_optional_array(
    const mlx_array handle) {
  return handle.ctx ? std::make_optional(shim_array_get(handle)) : std::nullopt;
}

mlx::core::fft::FFTNorm shim_fft_norm(const mlx_fft_norm norm) {
  switch (norm) {
    case MLX_FFT_NORM_BACKWARD:
      return mlx::core::fft::FFTNorm::Backward;
    case MLX_FFT_NORM_ORTHO:
      return mlx::core::fft::FFTNorm::Ortho;
    case MLX_FFT_NORM_FORWARD:
      return mlx::core::fft::FFTNorm::Forward;
  }
  throw std::invalid_argument("unknown FFT normalization");
}

mlx_dtype shim_dtype_to_c(const mlx::core::Dtype dtype) {
  switch (dtype.val()) {
    case mlx::core::Dtype::Val::bool_:
      return MLX_BOOL;
    case mlx::core::Dtype::Val::uint8:
      return MLX_UINT8;
    case mlx::core::Dtype::Val::uint16:
      return MLX_UINT16;
    case mlx::core::Dtype::Val::uint32:
      return MLX_UINT32;
    case mlx::core::Dtype::Val::uint64:
      return MLX_UINT64;
    case mlx::core::Dtype::Val::int8:
      return MLX_INT8;
    case mlx::core::Dtype::Val::int16:
      return MLX_INT16;
    case mlx::core::Dtype::Val::int32:
      return MLX_INT32;
    case mlx::core::Dtype::Val::int64:
      return MLX_INT64;
    case mlx::core::Dtype::Val::float16:
      return MLX_FLOAT16;
    case mlx::core::Dtype::Val::float32:
      return MLX_FLOAT32;
    case mlx::core::Dtype::Val::float64:
      return MLX_FLOAT64;
    case mlx::core::Dtype::Val::bfloat16:
      return MLX_BFLOAT16;
    case mlx::core::Dtype::Val::complex64:
      return MLX_COMPLEX64;
  }
  throw std::invalid_argument("unknown element type");
}

mlx::core::Dtype shim_dtype_from_c(const mlx_dtype dtype) {
  switch (dtype) {
    case MLX_BOOL:
      return mlx::core::bool_;
    case MLX_UINT8:
      return mlx::core::uint8;
    case MLX_UINT16:
      return mlx::core::uint16;
    case MLX_UINT32:
      return mlx::core::uint32;
    case MLX_UINT64:
      return mlx::core::uint64;
    case MLX_INT8:
      return mlx::core::int8;
    case MLX_INT16:
      return mlx::core::int16;
    case MLX_INT32:
      return mlx::core::int32;
    case MLX_INT64:
      return mlx::core::int64;
    case MLX_FLOAT16:
      return mlx::core::float16;
    case MLX_FLOAT32:
      return mlx::core::float32;
    case MLX_FLOAT64:
      return mlx::core::float64;
    case MLX_BFLOAT16:
      return mlx::core::bfloat16;
    case MLX_COMPLEX64:
      return mlx::core::complex64;
  }
  throw std::invalid_argument("unknown element type");
}

void shim_string_set(mlx_string* destination, const std::string& text) {
  mlx_string source = mlx_string_new_data(text.c_str());
  const int status = mlx_string_set(destination, source);
  mlx_string_free(source);
  if (status != 0) {
    throw std::runtime_error("could not copy the string result");
  }
}

void shim_vector_int_set(mlx_vector_int* destination, const std::vector<int>& values) {
  // MLX-C declares this copying parameter non-const but only reads from it.
  if (mlx_vector_int_set_data(
          destination, const_cast<int*>(values.data()), values.size()) != 0) {
    throw std::runtime_error("could not copy the integer vector result");
  }
}

} // namespace

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
