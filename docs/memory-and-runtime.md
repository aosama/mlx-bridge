# Memory and runtime

These functions apply to the whole process. The crate sets no limits of its own, so memory policy stays with your application.

## Counters

`active_memory_bytes` reports what arrays hold, `peak_memory_bytes` the high-water mark, and `cache_memory_bytes` what the allocator keeps for reuse. Because evaluation is lazy, active memory rises when a graph runs, not when it is built.

```rust
use mlx_c_rust::{MlxBindingsContext, MlxCError, MlxDtype, active_memory_bytes, install_non_terminating_error_handler, reset_peak_memory, peak_memory_bytes};

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
use mlx_c_rust::{MlxCError, install_non_terminating_error_handler, memory_limit_bytes, set_memory_limit};

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

## Releasing memory

Dropping every handle to an array releases it to the allocator's cache. `clear_allocator_cache` returns the cache to the system; call `synchronize` first so nothing in flight still uses it. In a long-running program, watch `active_memory_bytes` for leaks and `cache_memory_bytes` for growth.

## Printing

`MlxArray::to_string_lossy` renders an array as MLX prints it, and `set_print_format` controls precision and boolean spelling for the whole process. A negative precision selects MLX's default.

```rust,standalone_crate
use mlx_c_rust::{MlxArray, MlxCError, MlxPrintFormat, install_non_terminating_error_handler, print_format, set_print_format};

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

`MlxDtype::float_info` reports a floating-point type's limits, and fails for non-floating types.

```rust
use mlx_c_rust::{MlxCError, MlxDtype, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let limits = MlxDtype::Float32.float_info()?;
    assert_eq!(limits.bits, 32);
    assert_eq!(limits.eps, f64::from(f32::EPSILON));
    assert!(MlxDtype::Int32.float_info().is_err());
    Ok(())
}
```

## Platform queries

`metal_is_available`, `cuda_is_available`, and `distributed_is_available` report what this process can use, and `MlxBindingsContext::version` reports the linked MLX.
