/* The mlx-bridge gap shim's process-level entries: backend queries, the
 * memory limit check, and MLX's configuration values. These act on state
 * MLX keeps for the whole process or for the calling thread, not on arrays,
 * which is why they live apart from gap_shim.cpp. See gap_shim.h for the
 * contract and the retirement policy.
 */

#include <string>

#include "mlxrs/gap_shim.h"

#include "gap_shim_handles.h"
#include "mlx/allocator.h"
#include "mlx/c/error.h"
#include "mlx/config.h"

namespace {

const char* shim_config_name(const char* name) {
  if (name == nullptr) {
    throw std::invalid_argument("expected a configuration name");
  }
  return name;
}

} // namespace

extern "C" int mlxrs_is_nax_available(bool* res) {
  try {
    *res = mlx::core::metal::is_nax_available();
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_over_memory_limit(bool* res) {
  try {
    *res = mlx::core::allocator::over_memory_limit();
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_config_update(const char* name, int value) {
  try {
    mlx::core::config::update(shim_config_name(name), value);
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_config_get(int* res, const char* name, int default_value) {
  try {
    *res = mlx::core::config::get(shim_config_name(name), default_value);
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_config_scope_new(
    mlxrs_config_scope* res,
    const char* const* names,
    const int* values,
    size_t values_num) {
  try {
    mlx::core::config::ConfigDict scoped_values;
    for (size_t value_index = 0; value_index < values_num; ++value_index) {
      scoped_values[shim_config_name(names[value_index])] = values[value_index];
    }
    // The scope stays registered for exactly as long as this object lives;
    // freeing it is what removes the values from the thread.
    res->ctx = new mlx::core::config::ScopedUpdate(std::move(scoped_values));
  } catch (std::exception& error) {
    mlx_error("%s", error.what());
    return 1;
  }
  return 0;
}

extern "C" int mlxrs_config_scope_free(mlxrs_config_scope scope) {
  delete static_cast<mlx::core::config::ScopedUpdate*>(scope.ctx);
  return 0;
}
