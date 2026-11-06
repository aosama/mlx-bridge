# Errors

MLX-C reports failures through a process-wide handler, and its default handler ends the process. This crate replaces that with `MlxCError` values you can handle like any other Rust error.

## Install the handler first

Call `install_non_terminating_error_handler` before any fallible MLX call, once at the start of your program. Calling it again is harmless.

Without it, the first MLX failure prints a message and exits the process. This applies to every thread, and it is the most common cause of a program that dies with exit status 255.

## What an error contains

`MlxCError` has two public fields, and its `Display` output combines them:

- `operation`: what was attempted, such as `add MLX arrays`.
- `description`: what MLX reported, usually with the C source location that raised it.

```rust
use mlx_c_rust::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let left = MlxArray::from_f32(&[1.0, 2.0], &[2])?;
    let right = MlxArray::from_f32(&[1.0, 2.0, 3.0], &[3])?;

    let failure = context.add(&left, &right).expect_err("the shapes differ");

    assert_eq!(failure.operation, "add MLX arrays");
    assert!(failure.description.contains("broadcast"));
    assert!(failure.to_string().starts_with("MLX operation add MLX arrays failed:"));

    // A failure leaves nothing behind: the next call is unaffected.
    assert_eq!(context.add(&left, &left)?.to_vec_f32()?, vec![2.0, 4.0]);
    Ok(())
}
```

`MlxCError` implements `std::error::Error`, so `?` and error-handling crates work as usual.

## When errors arrive

| Failure | Arrives at |
|---|---|
| Incompatible shapes or element types | The operation call |
| Out-of-range axis | The operation call |
| Unsupported on this device | The operation call, or evaluation for some operations |
| Metal shader compilation | `evaluate` or a read |
| A singular matrix on the CPU | `evaluate` or a read |

Because of the last rows, a graph that built successfully can still fail when evaluated, so keep handling `Result` on `evaluate` and `to_vec_*`.

## Errors from your own closures

When a Rust callback fails inside an MLX transform, your original error comes back rather than MLX's generic "callback failed". A panic inside a callback is caught at the C boundary and returned as an error carrying the panic message, because unwinding across C is undefined behavior.

See [Closures and transforms](closures-and-transforms.md#failures-and-panics) for examples.

## Absence is not an error

Looking up a missing map key returns `Ok(None)`, and an exhausted map iterator returns `Ok(None)`. GGUF metadata probes such as `has_metadata_string` return `Ok(false)` for an absent key.

## Working with the channels directly

Most programs never need this, but code calling the `raw` layer can use the same machinery:

- `check_status(status, operation)` turns an MLX-C status code into a `Result`, preferring a parked callback failure as the root cause.
- `take_captured_mlx_error` and `clear_captured_mlx_error` read and discard this thread's last MLX message.
- `mlx_c_rust::error::set_closure_error`, `take_closure_error`, and `clear_closure_error` manage the slot a callback parks its failure in.

```rust
use mlx_c_rust::{check_status, clear_captured_mlx_error, install_non_terminating_error_handler};

install_non_terminating_error_handler();
clear_captured_mlx_error();

assert!(check_status(0, "a successful call").is_ok());

let failure = check_status(7, "a silent failure").expect_err("nonzero is a failure");
assert_eq!(failure.operation, "a silent failure");
assert!(failure.description.contains("status 7"));
```

## Calls that cannot report an error

A few cheap queries return a plain value. `MlxArray::dimension` returns 0 for an out-of-range axis, which looks the same as a zero-sized dimension, so check against `shape().len()` when the axis is not known to be valid. On an empty handle, `shape`, `dtype`, `element_count`, and `byte_count` return empty values, and they go through MLX-C's error path, so install the handler first and check `is_empty` when a handle might be empty.
