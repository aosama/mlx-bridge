# Evaluation, streams, and devices

Three questions decide how your program behaves: when does work run, where does it run, and on which thread can you use it?

## Lazy evaluation

Operations only record a graph. MLX computes values when you evaluate an array, read its values, or use it in something that forces evaluation.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let values = MlxArray::from_f32(&[1.0, 2.0], &[2])?;

    let doubled = context.add(&values, &values)?;
    assert!(!doubled.is_available()?, "nothing has run yet");

    doubled.evaluate()?;
    assert!(doubled.is_available()?);
    assert_eq!(doubled.to_vec_f32()?, vec![2.0, 4.0]);
    Ok(())
}
```

Two consequences matter in practice:

- **Some errors arrive late.** Shape and type errors usually surface when the graph is built, but failures such as a Metal shader that does not compile surface at `evaluate`. Handle errors from both.
- **Timing the wrong call measures nothing.** A loop of operations can return instantly and then do all its work at the first read. Evaluate explicitly before you stop the clock.

## Evaluating several arrays or in the background

`evaluate_arrays` evaluates several arrays together, which lets them share work. `async_eval_arrays` starts evaluation and returns, and `wait` blocks until a value is ready.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler, synchronize};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let values = MlxArray::from_f32(&[1.0, 2.0], &[2])?;
    let doubled = context.add(&values, &values)?;
    let tripled = context.multiply_scalar(&values, 3.0)?;

    context.evaluate_arrays(&[&doubled, &tripled])?;

    let squared = context.square(&values)?;
    context.async_eval_arrays(&[&squared])?;
    squared.wait()?;

    // Waits for everything submitted to this stream so far.
    synchronize(context.stream())?;
    assert_eq!(squared.to_vec_f32()?, vec![1.0, 4.0]);
    Ok(())
}
```

`wait` is always safe: on an array nothing has scheduled yet, it schedules it first. This is deliberately different from MLX-C's `_mlx_array_wait`, which would mark such an array available with no data behind it.

## Choosing where work runs

Every context submits to one stream, and a stream belongs to one device.

| Constructor | Runs on | Use it for |
|---|---|---|
| `MlxBindingsContext::new()` | The default GPU stream | Almost everything |
| `MlxBindingsContext::on_cpu()` | The default CPU stream | Linear algebra, `float64` |
| `MlxBindingsContext::on_stream(stream)` | A stream you created | Separating workloads |

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, MlxDeviceHandle, MlxStream, install_non_terminating_error_handler, raw};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let gpu = MlxDeviceHandle::of_type(raw::mlx_device_type__MLX_GPU, 0)?;
    let custom_stream = MlxStream::on_device(&gpu)?;
    let context = MlxBindingsContext::on_stream(custom_stream)?;

    let values = MlxArray::from_f32(&[1.0], &[1])?;
    assert_eq!(context.add(&values, &values)?.to_vec_f32()?, vec![2.0]);
    assert!(context.stream().device()?.equals(&gpu));
    Ok(())
}
```

Arrays are not tied to a device. Apple silicon memory is unified, so an array built on one stream can be used by an operation on another without copying.

## What the GPU cannot do

These fail with a clear error on a GPU context, so use `on_cpu`:

- `float64` arrays: `astype` to `Float64` fails when the graph is built.
- Matrix factorization and solving: `linalg_inv`, `linalg_solve`, `linalg_cholesky`, `linalg_qr`, `linalg_svd`, and similar. See [Linear algebra](linear-algebra.md).

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let values = MlxArray::from_f32(&[1.5], &[1])?;

    let gpu = MlxBindingsContext::new()?;
    assert!(gpu.astype(&values, MlxDtype::Float64).is_err());

    let cpu = MlxBindingsContext::on_cpu()?;
    assert_eq!(cpu.astype(&values, MlxDtype::Float64)?.to_vec_f64()?, vec![1.5]);
    Ok(())
}
```

## Streams

- `MlxStream::default_gpu` and `default_cpu` return this thread's defaults.
- `MlxStream::on_device` creates a new stream.
- `MlxStream::default_for_device` and `set_default_stream` read and replace a device's default.
- `MlxStream::peek_default_for_device` returns the default only if it already exists, without creating it.
- `MlxStream::all_streams` lists the streams created so far.

## Devices

`MlxDeviceHandle` identifies a device and answers questions about it.

```rust
use mlx_bridge::{MlxCError, MlxDeviceHandle, install_non_terminating_error_handler, raw};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let gpu = MlxDeviceHandle::of_type(raw::mlx_device_type__MLX_GPU, 0)?;
    assert!(gpu.is_available()?);

    let information = gpu.information()?;
    assert!(information.has_key("device_name")?);
    assert!(!information.string_value("device_name")?.is_empty());
    assert!(information.size_value("memory_size")? > 0);
    Ok(())
}
```

## Threads

Streams are thread affine: evaluating on a stream from a thread other than the one that created it fails. `MlxArray` and `MlxStream` are not `Send`, so the compiler enforces this.

Each thread should create its own context:

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn double_on_this_thread(values: &[f32]) -> Result<Vec<f32>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    let row_length = i32::try_from(values.len()).expect("the fixture is small");
    let array = MlxArray::from_f32(values, &[row_length])?;
    context.add(&array, &array)?.to_vec_f32()
}

fn double_fixture() -> Result<Vec<f32>, MlxCError> {
    double_on_this_thread(&[1.0, 2.0])
}

fn main() {
    install_non_terminating_error_handler();
    let worker = std::thread::spawn(double_fixture);
    let doubled = worker.join().expect("the worker should finish");
    assert_eq!(doubled.expect("the computation should work"), vec![2.0, 4.0]);
}
```

To move values between threads, copy them out with `to_vec_*` and build a new array on the other side. Errors are captured per thread, so a failure on one thread never surfaces on another.

`MlxStream::new_thread_unsafe` creates a stream usable from any thread, but MLX applies no synchronization to it, so it is an `unsafe fn` and you must rule out data races yourself.

## Related

- [Memory and runtime](memory-and-runtime.md) covers memory limits and counters.
- The [runtime reference](reference/runtime.md) lists every stream function.
