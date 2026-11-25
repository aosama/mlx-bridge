# Memory and runtime

These functions apply to the whole process. The crate sets no limits of its own, so memory policy stays with your application.

## Counters

`active_memory_bytes` reports what arrays hold, `peak_memory_bytes` the high-water mark, and `cache_memory_bytes` what the allocator keeps for reuse. Because evaluation is lazy, active memory rises when a graph runs, not when it is built.

```rust
use mlx_bridge::{MlxBindingsContext, MlxCError, MlxDtype, active_memory_bytes, install_non_terminating_error_handler, reset_peak_memory, peak_memory_bytes};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    reset_peak_memory()?;

    let values = context.full(&[1024 * 1024], 1.0, MlxDtype::Float32)?;
    values.evaluate()?;

    assert!(active_memory_bytes()? >= 4 * 1024 * 1024);
    assert!(peak_memory_bytes()? >= 4 * 1024 * 1024);
    Ok(())
}
```

`array_buffer_size_bytes` reports the bytes behind particular arrays rather than the whole process.

## Limits

`set_memory_limit` and `set_cache_limit` return the previous value, so you can restore it. `set_wired_limit` controls how much memory the GPU keeps wired.

```rust,standalone_crate
use mlx_bridge::{MlxCError, install_non_terminating_error_handler, memory_limit_bytes, set_memory_limit};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let original_limit = memory_limit_bytes()?;

    let previous_limit = set_memory_limit(original_limit / 2)?;
    assert_eq!(previous_limit, original_limit);
    assert_eq!(memory_limit_bytes()?, original_limit / 2);

    set_memory_limit(original_limit)?;
    Ok(())
}
```

`is_over_memory_limit` reports whether active memory is above the limit. While it is and work is in flight, MLX holds back new evaluations until memory drains, so check it before queueing a large graph.

## Configuration

MLX reads integer tunables such as `MLX_MAX_OPS_PER_BUFFER` from the environment. `config_update` sets one for the whole process and `config_get` reads one back, falling back to the environment and then to the default you pass. `MlxConfigScope` applies values on the current thread until it drops, ahead of process values. A value takes effect where MLX reads the name after the change, and MLX reads some names only once, for example when it first creates a device or stream, so set those before the first computation. The first read of a name records its answer, so a later read with another default returns the first answer.

```rust
use mlx_bridge::{MlxCError, MlxConfigScope, config_get, config_update, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    config_update("MLX_MAX_OPS_PER_BUFFER", 40)?;
    assert_eq!(config_get("MLX_MAX_OPS_PER_BUFFER", 10)?, 40);

    let scope = MlxConfigScope::new(&[("MLX_MAX_OPS_PER_BUFFER", 20)])?;
    assert_eq!(config_get("MLX_MAX_OPS_PER_BUFFER", 10)?, 20);
    drop(scope);
    assert_eq!(config_get("MLX_MAX_OPS_PER_BUFFER", 10)?, 40);
    Ok(())
}
```

A scope belongs to the thread that created it, so `MlxConfigScope` is neither `Send` nor `Sync`.

## Releasing memory

Dropping every handle to an array releases it to the allocator's cache. `clear_allocator_cache` returns the cache to the system; call `synchronize` first so nothing in flight still uses it. In a long-running program, watch `active_memory_bytes` for leaks and `cache_memory_bytes` for growth.

## Printing

`MlxArray::to_string_lossy` renders an array as MLX prints it, and `set_print_format` controls precision and boolean spelling for the whole process. A negative precision selects MLX's default.

```rust,standalone_crate
use mlx_bridge::{MlxArray, MlxCError, MlxPrintFormat, install_non_terminating_error_handler, print_format, set_print_format};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let original_format = print_format()?;

    set_print_format(MlxPrintFormat { precision: 2, capitalize_bool: true })?;
    let rendered = MlxArray::from_f32(&[1.23456], &[1])?.to_string_lossy()?;
    assert!(rendered.contains("1.23"));
    assert!(!rendered.contains("1.234"));

    set_print_format(original_format)?;
    Ok(())
}
```

## Element type information

`MlxDtype::float_info` reports a floating-point type's limits, and fails for non-floating types. `promote_with` gives the type MLX assigns the result of combining two types, `kind` gives MLX's coarse classification, and `is_in_category` tests a type against MLX's hierarchy of categories.

```rust
use mlx_bridge::{MlxCError, MlxDtype, MlxDtypeCategory, MlxDtypeKind, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let limits = MlxDtype::Float32.float_info()?;
    assert_eq!(limits.bits, 32);
    assert_eq!(limits.eps, f64::from(f32::EPSILON));
    assert!(MlxDtype::Int32.float_info().is_err());

    assert_eq!(MlxDtype::Int8.promote_with(MlxDtype::UInt8)?, MlxDtype::Int16);
    assert_eq!(MlxDtype::BFloat16.kind()?, MlxDtypeKind::BrainFloating);
    assert!(MlxDtype::BFloat16.is_in_category(MlxDtypeCategory::Floating)?);
    assert!(!MlxDtype::Bool.is_in_category(MlxDtypeCategory::Number)?);
    Ok(())
}
```

## Platform queries

`metal_is_available`, `cuda_is_available`, and `distributed_is_available` report what this process can use, `metal_is_nax_available` reports whether MLX's Metal kernels can use the GPU neural accelerators on this machine, and `MlxBindingsContext::version` reports the linked MLX.
