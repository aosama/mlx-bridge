# Arrays

`MlxArray` is an owned handle to an MLX array. This guide covers creating arrays, reading them, sharing them, and how views behave.

## Creating arrays from host values

There is one constructor per stored element type. Values are given in row-major order and copied at the call, so later changes to your slice do not affect the array.

| Element type | Constructor | Reader |
|---|---|---|
| `Bool` | `from_bools` | `to_vec_bool` |
| `UInt8`, `UInt16`, `UInt32`, `UInt64` | `from_u8`, `from_u16`, `from_u32`, `from_u64` | `to_vec_u8`, `to_vec_u16`, `to_vec_u32`, `to_vec_u64` |
| `Int8`, `Int16`, `Int32`, `Int64` | `from_i8`, `from_i16`, `from_i32`, `from_i64` | `to_vec_i8`, `to_vec_i16`, `to_vec_i32`, `to_vec_i64` |
| `Float32`, `Float64` | `from_f32`, `from_f64` | `to_vec_f32`, `to_vec_f64` |
| `Float16`, `BFloat16` | `from_f16_bits`, `from_bf16_bits` (raw bits) | `to_vec_f16_bits`, `to_vec_bf16_bits` |
| `Complex64` | `from_complex` (`(real, imaginary)` pairs) | `to_vec_complex` |

```rust
use mlx_bridge::{MlxArray, MlxCError, MlxDtype};

fn main() -> Result<(), MlxCError> {
    let matrix = MlxArray::from_i64(&[1, 2, 3, 4, 5, 6], &[2, 3])?;

    assert_eq!(matrix.shape(), vec![2, 3]);
    assert_eq!(matrix.dtype(), MlxDtype::Int64);
    assert_eq!(matrix.element_count(), 6);
    assert_eq!(matrix.byte_count(), 48);
    assert_eq!(matrix.to_vec_i64()?, vec![1, 2, 3, 4, 5, 6]);
    Ok(())
}
```

A shape whose element count differs from the number of values returns an error rather than reading out of bounds, and so does a negative dimension.

```rust
use mlx_bridge::MlxArray;

assert!(MlxArray::from_f32(&[1.0, 2.0, 3.0], &[2, 2]).is_err());
assert!(MlxArray::from_i32(&[1, 2], &[-2]).is_err());
```

## Scalars and zero-sized arrays

An empty shape means a zero-dimensional scalar holding one value. A shape containing zero is a real array with no elements.

```rust
use mlx_bridge::{MlxArray, MlxCError};

fn main() -> Result<(), MlxCError> {
    let scalar = MlxArray::from_f32(&[2.5], &[])?;
    assert_eq!(scalar.shape(), Vec::<i32>::new());
    assert_eq!(scalar.element_count(), 1);
    assert_eq!(scalar.item_f32()?, 2.5);

    let zero_sized = MlxArray::from_f32(&[], &[0])?;
    assert!(!zero_sized.is_empty(), "zero-sized is not an empty handle");
    assert_eq!(zero_sized.to_vec_f32()?, Vec::<f32>::new());
    Ok(())
}
```

Shortcuts exist for common scalars: `from_bool`, `from_scalar_i32`, `from_scalar_f32`, `from_scalar_f64`, and `from_scalar_complex`.

## Creating arrays with operations

Filled and generated arrays come from a context, because they run on a stream.

```rust
use mlx_bridge::{MlxBindingsContext, MlxCError, MlxDtype, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;

    assert_eq!(context.zeros(&[2], MlxDtype::Int32)?.to_vec_i32()?, vec![0, 0]);
    assert_eq!(context.ones(&[2], MlxDtype::Float32)?.to_vec_f32()?, vec![1.0, 1.0]);
    assert_eq!(context.full(&[2], 2.5, MlxDtype::Float32)?.to_vec_f32()?, vec![2.5, 2.5]);
    assert_eq!(context.arange_i32(2, 5)?.to_vec_i32()?, vec![2, 3, 4]);
    assert_eq!(context.arange_f32(0.0, 1.0, 0.25)?.to_vec_f32()?, vec![0.0, 0.25, 0.5, 0.75]);
    assert_eq!(context.eye(2, 2, 0, MlxDtype::Int32)?.to_vec_i32()?, vec![1, 0, 0, 1]);
    Ok(())
}
```

## Reading values back

Each `to_vec_*` reader evaluates the array and copies its elements into a new `Vec`. The `Vec` is yours: changing it does not touch the array.

### The element type must match

MLX's raw accessors reinterpret memory, so this crate checks the element type first. A mismatch is an error, not a conversion. Use `astype` when you want a conversion.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let values = MlxArray::from_f32(&[1.5], &[1])?;

    assert!(values.to_vec_i32().is_err());
    assert!(values.item_u32().is_err());

    let converted = context.astype(&values, MlxDtype::Int32)?;
    assert_eq!(converted.to_vec_i32()?, vec![1]);
    Ok(())
}
```

### Views read in logical order

Transposes, reversals, and strided slices are views over their parent's storage. Readers materialize them first when needed, so you always get logical row-major order.

```rust
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, install_non_terminating_error_handler};

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let context = MlxBindingsContext::new()?;
    let matrix = MlxArray::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

    let transposed = context.transpose(&matrix)?;
    assert_eq!(transposed.shape(), vec![3, 2]);
    assert_eq!(transposed.to_vec_f32()?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert!(!transposed.is_row_contiguous()?);

    let row = MlxArray::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4])?;
    assert_eq!(context.flip(&row)?.to_vec_f32()?, vec![4.0, 3.0, 2.0, 1.0]);
    assert_eq!(context.slice(&row, &[0], &[4], &[2])?.to_vec_f32()?, vec![1.0, 3.0]);
    Ok(())
}
```

`copy_evaluated_u8_values` and `copy_evaluated_u32_values` are the exception: they exist to measure one direct host transfer, so they return an error for a view rather than materializing it. Use `to_vec_u8`, `to_vec_u32`, or `MlxBindingsContext::copy_u32_values` if you just want the values.

### Scalars

`item_u8` through `item_f64`, `item_bool`, `item_complex`, `item_f16_bits`, and `item_bf16_bits` read a single element. They require the array to hold exactly one element of exactly that type.

## Sharing and ownership

Handles are move-only. `retain` gives you a second handle to the same array without copying its data, and each handle keeps the array alive independently.

```rust
use mlx_bridge::{MlxArray, MlxCError};

fn main() -> Result<(), MlxCError> {
    let source = MlxArray::from_f32(&[1.0, 2.0], &[2])?;
    let mut retained = source.retain()?;
    drop(source);
    assert_eq!(retained.to_vec_f32()?, vec![1.0, 2.0]);

    // Setters replace what this one handle refers to, never what others see.
    let original = retained.retain()?;
    retained.set_f32(9.0)?;
    assert_eq!(original.to_vec_f32()?, vec![1.0, 2.0]);
    assert_eq!(retained.item_f32()?, 9.0);
    Ok(())
}
```

Arrays also outlive the context and the inputs that built them, because the graph holds what it needs.

## Empty handles

`MlxArray::empty()` holds no array. It exists mainly as an output parameter for `raw` calls. Evaluating it or reading its values returns an error, and `is_empty` reports it. The plain queries `shape`, `dtype`, `element_count`, and `byte_count` cannot report failure, so check `is_empty` before trusting them on a handle that might be empty. A zero-sized array is not empty in this sense.

## Other array containers

- `MlxArrayVector` holds an ordered list of array handles, used by closures and multi-output operations.
- `MlxVectorVectorArray` nests those lists.
- `MlxMapStringToArray` maps names to arrays, used by keyword arguments and safetensors.

```rust
use mlx_bridge::{MlxArray, MlxArrayVector, MlxCError, MlxMapStringToArray};

fn main() -> Result<(), MlxCError> {
    let weights = MlxArray::from_f32(&[1.0], &[1])?;
    let vector = MlxArrayVector::new(&[&weights, &weights])?;
    assert_eq!(vector.len(), 2);

    let mut map = MlxMapStringToArray::new()?;
    map.insert("weights", &weights)?;
    assert!(map.value_at("weights")?.is_some());
    assert!(map.value_at("missing")?.is_none());

    let mut iterator = map.iter()?;
    while let Some((name, _array)) = iterator.next_entry()? {
        assert_eq!(name, "weights");
    }
    Ok(())
}
```

The map's iterator borrows the map, so the compiler stops you from dropping the map while iterating.

## Related

- [Evaluation, streams, and devices](evaluation-and-streams.md) explains when values are computed.
- The [types reference](reference/types.md) lists every method.
