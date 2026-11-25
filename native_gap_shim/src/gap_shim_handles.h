/* Handle and value conversions shared by the gap shim's sources. Private to
 * the shim: it is not installed, and consumers include only gap_shim.h.
 *
 * The handle casts inline what MLX-C's private headers do: the public C
 * struct carries the C++ object as a void* ctx (that layout is the
 * published C ABI), so casting ctx is self-contained, needs no MLX-C private
 * header, and is exactly what the vendored MLX-C itself does. Every helper
 * is inline because each shim source includes this header.
 */

#ifndef MLXRS_GAP_SHIM_HANDLES_H
#define MLXRS_GAP_SHIM_HANDLES_H

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mlx/c/mlx.h"
#include "mlx/mlx.h"
// For the shim's own C enums, which the kind and category conversions map.
#include "mlxrs/gap_shim.h"

namespace mlxrs_shim {

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

inline mlx::core::fft::FFTNorm shim_fft_norm(const mlx_fft_norm norm) {
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

inline mlx_dtype shim_dtype_to_c(const mlx::core::Dtype dtype) {
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

inline mlx::core::Dtype shim_dtype_from_c(const mlx_dtype dtype) {
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

inline mlxrs_dtype_kind shim_dtype_kind_to_c(const mlx::core::Dtype::Kind kind) {
  switch (kind) {
    case mlx::core::Dtype::Kind::b:
      return MLXRS_DTYPE_KIND_BOOL;
    case mlx::core::Dtype::Kind::u:
      return MLXRS_DTYPE_KIND_UNSIGNED_INTEGER;
    case mlx::core::Dtype::Kind::i:
      return MLXRS_DTYPE_KIND_SIGNED_INTEGER;
    case mlx::core::Dtype::Kind::f:
      return MLXRS_DTYPE_KIND_FLOATING;
    case mlx::core::Dtype::Kind::c:
      return MLXRS_DTYPE_KIND_COMPLEX;
    case mlx::core::Dtype::Kind::V:
      return MLXRS_DTYPE_KIND_BRAIN_FLOATING;
  }
  throw std::invalid_argument("unknown element type kind");
}

inline mlx::core::Dtype::Category shim_dtype_category_from_c(
    const mlxrs_dtype_category category) {
  switch (category) {
    case MLXRS_DTYPE_CATEGORY_COMPLEX_FLOATING:
      return mlx::core::Dtype::Category::complexfloating;
    case MLXRS_DTYPE_CATEGORY_FLOATING:
      return mlx::core::Dtype::Category::floating;
    case MLXRS_DTYPE_CATEGORY_INEXACT:
      return mlx::core::Dtype::Category::inexact;
    case MLXRS_DTYPE_CATEGORY_SIGNED_INTEGER:
      return mlx::core::Dtype::Category::signedinteger;
    case MLXRS_DTYPE_CATEGORY_UNSIGNED_INTEGER:
      return mlx::core::Dtype::Category::unsignedinteger;
    case MLXRS_DTYPE_CATEGORY_INTEGER:
      return mlx::core::Dtype::Category::integer;
    case MLXRS_DTYPE_CATEGORY_NUMBER:
      return mlx::core::Dtype::Category::number;
    case MLXRS_DTYPE_CATEGORY_GENERIC:
      return mlx::core::Dtype::Category::generic;
  }
  throw std::invalid_argument("unknown element type category");
}

inline void shim_string_set(mlx_string* destination, const std::string& text) {
  mlx_string source = mlx_string_new_data(text.c_str());
  const int status = mlx_string_set(destination, source);
  mlx_string_free(source);
  if (status != 0) {
    throw std::runtime_error("could not copy the string result");
  }
}

inline void shim_vector_int_set(mlx_vector_int* destination, const std::vector<int>& values) {
  // MLX-C declares this copying parameter non-const but only reads from it.
  if (mlx_vector_int_set_data(
          destination, const_cast<int*>(values.data()), values.size()) != 0) {
    throw std::runtime_error("could not copy the integer vector result");
  }
}


} // namespace mlxrs_shim

#endif /* MLXRS_GAP_SHIM_HANDLES_H */
