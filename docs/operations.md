# Operations

Operations are methods on `MlxBindingsContext`. This guide covers the conventions they share, then where each family lives. For what an operation means mathematically, MLX's own [documentation](https://ml-explore.github.io/mlx/) is the authority; the behavior here is the same.

## Conventions

- **The context chooses the stream.** Each call submits to the context's stream. See [choosing where work runs](evaluation-and-streams.md#choosing-where-work-runs).
- **Inputs are borrowed.** Operations take `&MlxArray` and never consume their inputs.
- **Results are lazy and owned.** You get a new `MlxArray` holding an unevaluated graph.
- **Negative axes count from the end**, so `-1` is the last axis.
- **`Option` means MLX's default.** Passing `None` selects the default for that parameter.

## Naming patterns

MLX's overloads become separate methods, usually with a suffix. A few differ, so check the signature: `cumsum` takes an axis, and its flattened form is `cumsum_across_all_axes`.

| Form | Meaning | Example |
|---|---|---|
| No suffix | Usually every axis, flattening where relevant | `sum`, `sort`, `argmax` |
| `_axis` | One axis | `sum_axis`, `sort_axis` |
| `_axes` | Several axes | `sum_axes`, `max_axes` |

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let values = MlxArray::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

    assert_eq!(context.sum(&values, false)?.item_f32()?, 21.0);
    assert_eq!(context.sum_axis(&values, 1, false)?.to_vec_f32()?, vec![6.0, 15.0]);
    assert_eq!(context.sum_axis(&values, -2, false)?.to_vec_f32()?, vec![5.0, 7.0, 9.0]);

    // keepdims keeps the reduced axis with length one.
    assert_eq!(context.sum_axis(&values, 0, true)?.shape(), vec![1, 3]);
    assert_eq!(context.sum_axes(&values, &[0, 1], false)?.shape(), Vec::<i32>::new());
    Ok(())
}
```

## Broadcasting and element types

Binary operations broadcast their inputs and promote element types as MLX does. To ask MLX the answer without running anything, use `broadcast_shapes` and `result_type`.

```rust
use mlx_bridge::{MlxArray, MlxArrayVector, MlxCError, MlxDtype, broadcast_shapes, install_non_terminating_error_handler, result_type};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    assert_eq!(broadcast_shapes(&[2, 1], &[3])?, vec![2, 3]);
    assert!(broadcast_shapes(&[2], &[3]).is_err());

    let integers = MlxArray::from_i32(&[1], &[1])?;
    let floats = MlxArray::from_f32(&[1.0], &[1])?;
    let operands = MlxArrayVector::new(&[&integers, &floats])?;
    assert_eq!(result_type(&operands)?, MlxDtype::Float32);
    Ok(())
}
```

## Shape and indexing

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let values = MlxArray::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6])?;

    // -1 infers one dimension from the element count.
    let matrix = context.reshape(&values, &[2, -1])?;
    assert_eq!(matrix.shape(), vec![2, 3]);
    assert_eq!(context.expand_dims(&values, 0)?.shape(), vec![1, 6]);

    let slice = context.slice(&values, &[1], &[3], &[1])?;
    assert_eq!(slice.to_vec_f32()?, vec![2.0, 3.0]);

    let update = MlxArray::from_f32(&[8.0, 9.0], &[2])?;
    let updated = context.slice_update(&values, &update, &[0], &[2], &[1])?;
    assert_eq!(updated.to_vec_f32()?, vec![8.0, 9.0, 3.0, 4.0, 5.0, 6.0]);

    let indices = MlxArray::from_i32(&[1, 0], &[2])?;
    let row_order = context.take_axis(&matrix, &indices, 0)?;
    assert_eq!(row_order.to_vec_f32()?, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);

    let parts = context.split(&values, 2, 0)?;
    assert_eq!(parts.array_at(1, "read the second half")?.to_vec_f32()?, vec![4.0, 5.0, 6.0]);
    Ok(())
}
```

## Choosing values

`where_select` picks from two arrays by a boolean condition, and comparisons such as `greater` produce that condition.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let left = MlxArray::from_f32(&[1.0, 2.0], &[2])?;
    let right = MlxArray::from_f32(&[3.0, 1.0], &[2])?;

    let is_left_larger = context.greater(&left, &right)?;
    let largest = context.where_select(&is_left_larger, &left, &right)?;

    assert_eq!(largest.to_vec_f32()?, vec![3.0, 2.0]);
    Ok(())
}
```

## Reductions and cumulative operations

`keepdims` is explicit on most reductions, though `argmax_axis` has none, and `argmin_axis` does. `std` and `var` also take `ddof`. Cumulative operations take `reverse` and `inclusive`.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let values = MlxArray::from_f32(&[1.0, 2.0, 3.0], &[3])?;

    assert_eq!(context.cumsum(&values, 0, false, true)?.to_vec_f32()?, vec![1.0, 3.0, 6.0]);
    assert_eq!(context.cumsum(&values, 0, true, true)?.to_vec_f32()?, vec![6.0, 5.0, 3.0]);
    assert_eq!(context.cumsum(&values, 0, false, false)?.to_vec_f32()?, vec![0.0, 1.0, 3.0]);
    assert_eq!(context.argmax(&values, false)?.item_u32()?, 2);
    Ok(())
}
```

## Sorting and selection

`sort` and `argsort` work over the flattened array, and the `_axis` forms along one axis. `topk` returns the largest values in no guaranteed order, and `partition` places the element at `kth` in its sorted position.

## Where each family lives

| Family | Examples | Where to read more |
|---|---|---|
| Arithmetic and comparison | `add`, `multiply`, `greater`, `where_select`, `matmul` | `src/operations/general.rs` |
| Binary math | `equal`, `maximum`, `remainder`, `bitwise_and`, `isclose`, `inner`, `outer` | `src/operations/math_binary.rs` |
| Unary math | `abs`, `log`, `round`, `isnan`, `sort`, `nan_to_num` | `src/operations/math_unary.rs` |
| Creation | `zeros`, `full`, `arange_i32`, `eye`, `linspace`, `ones_like` | `src/operations/creation.rs`, `array_creation.rs` |
| Shape | `reshape`, `transpose`, `concatenate_axis`, `split`, `flatten`, `unique` | `src/operations/shape.rs`, `array_shape.rs` |
| Reduction | `sum`, `mean`, `max`, `var`, `argmax`, `all`, `any` | `src/operations/reduction.rs` |
| Cumulative | `cummax`, `cumprod`, `logcumsumexp` | `src/operations/cumulative.rs` |
| Indexing | `take`, `gather`, `scatter`, `einsum`, `tensordot`, `searchsorted` | `src/operations/indexing.rs` |
| Activation | `silu`, `softplus`, `gelu`, `tanh`, and `sigmoid` in `general.rs` | `src/operations/activation.rs` |
| Linear algebra | `linalg_inv`, `linalg_qr`, `linalg_norm` | [guide](linear-algebra.md) |
| FFT and STFT | `fft_rfft`, `stft`, `istft` | [guide](signal-processing.md) |
| Random | `random_uniform`, `random_normal`, `random_key` | [guide](random.md) |
| Model primitives | attention, RoPE, convolution, quantization | [guide](model-primitives.md) |

Every method is listed in the [operations reference](reference/operations.md).
