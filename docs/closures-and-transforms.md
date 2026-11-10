# Closures and transforms

MLX's transforms take functions and return new ones. In this crate those functions are `MlxClosure` values built from ordinary Rust callables, so gradients, batching, and compilation all work on Rust code.

## Building a closure

`MlxClosure::from_fn` accepts any callable taking `&[MlxArray]` and returning `Result<Vec<MlxArray>, MlxCError>`. A named function is the simplest choice. Build a context inside the callback when it needs to run operations.

```rust
use mlx_bridge::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxClosure, install_non_terminating_error_handler};

fn doubled(inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    Ok(vec![context.multiply_scalar(&inputs[0], 2.0)?])
}

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let closure = MlxClosure::from_fn(doubled)?;
    let input = MlxArray::from_f32(&[1.0, 2.0], &[2])?;

    let outputs = closure.apply(&MlxArrayVector::from_array(&input)?)?;

    assert_eq!(outputs.array_at(0, "read the output")?.to_vec_f32()?, vec![2.0, 4.0]);
    Ok(())
}
```

The closure owns the callable and releases it when the last handle to it drops. Copies made with `set`, and closures derived through transforms, keep it alive.

## Gradients

`value_and_grad_of_closure` differentiates with respect to the argument positions you name, and the function must return a scalar first output.

```rust
use mlx_bridge::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxClosure, install_non_terminating_error_handler, value_and_grad_of_closure};

fn sum_of_squares(inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    Ok(vec![context.sum(&context.square(&inputs[0])?, false)?])
}

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let function = MlxClosure::from_fn(sum_of_squares)?;
    let value_and_gradient = value_and_grad_of_closure(&function, &[0])?;
    let point = MlxArray::from_f32(&[3.0], &[1])?;

    let (values, gradients) = value_and_gradient.apply(&MlxArrayVector::from_array(&point)?)?;

    assert_eq!(values.array_at(0, "read the value")?.item_f32()?, 9.0);
    assert_eq!(gradients.array_at(0, "read the gradient")?.to_vec_f32()?, vec![6.0]);
    Ok(())
}
```

## Jacobian products

`vjp_of_closure` and `jvp_of_closure` take the primals and the cotangents or tangents, and return the outputs with the products.

```rust
use mlx_bridge::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxClosure, install_non_terminating_error_handler, jvp_of_closure, vjp_of_closure};

fn sum_of_squares(inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    Ok(vec![context.sum(&context.square(&inputs[0])?, false)?])
}

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let function = MlxClosure::from_fn(sum_of_squares)?;
    let primals = MlxArrayVector::from_array(&MlxArray::from_f32(&[3.0], &[1])?)?;

    let cotangents = MlxArrayVector::from_array(&MlxArray::from_scalar_f32(1.0))?;
    let (_outputs, pulled_back) = vjp_of_closure(&function, &primals, &cotangents)?;
    assert_eq!(pulled_back.array_at(0, "read")?.to_vec_f32()?, vec![6.0]);

    let tangents = MlxArrayVector::from_array(&MlxArray::from_f32(&[1.0], &[1])?)?;
    let (_outputs, pushed_forward) = jvp_of_closure(&function, &primals, &tangents)?;
    assert_eq!(pushed_forward.array_at(0, "read")?.item_f32()?, 6.0);
    Ok(())
}
```

## Batching with `vmapped`

`MlxClosure::vmapped` vectorizes a closure: inputs map over `input_axes`, and outputs stack along `output_axes`. An empty list means axis 0 for everything.

```rust
use mlx_bridge::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxClosure, install_non_terminating_error_handler};

fn doubled(inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    Ok(vec![context.multiply_scalar(&inputs[0], 2.0)?])
}

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let function = MlxClosure::from_fn(doubled)?;
    let batch = MlxArray::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let inputs = MlxArrayVector::from_array(&batch)?;

    let over_rows = function.vmapped(&[], &[])?.apply(&inputs)?;
    assert_eq!(over_rows.array_at(0, "read")?.shape(), vec![2, 3]);

    // Map over the columns, then stack those results first.
    let over_columns = function.vmapped(&[1], &[0])?.apply(&inputs)?;
    assert_eq!(over_columns.array_at(0, "read")?.shape(), vec![3, 2]);
    Ok(())
}
```

`vmap_trace` and `vmap_replace` expose the two building blocks for custom batching rules. Most programs want `vmapped`.

## Compilation

`compiled` wraps a closure so MLX traces it once and replays an optimized graph. Results match the eager path. With `shapeless` set to `true`, one compilation serves any input shape, which only works for graphs whose operations do not depend on the shape.

```rust
use mlx_bridge::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxClosure, install_non_terminating_error_handler};

fn doubled(inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    Ok(vec![context.multiply_scalar(&inputs[0], 2.0)?])
}

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let compiled = MlxClosure::from_fn(doubled)?.compiled(true)?;

    let short_input = MlxArray::from_f32(&[1.0], &[1])?;
    let long_input = MlxArray::from_f32(&[1.0, 2.0, 3.0], &[3])?;

    let short_outputs = compiled.apply(&MlxArrayVector::from_array(&short_input)?)?;
    let long_outputs = compiled.apply(&MlxArrayVector::from_array(&long_input)?)?;

    assert_eq!(short_outputs.array_at(0, "read")?.to_vec_f32()?, vec![2.0]);
    assert_eq!(long_outputs.array_at(0, "read")?.to_vec_f32()?, vec![2.0, 4.0, 6.0]);
    Ok(())
}
```

`disable_compile`, `enable_compile`, and `set_compile_mode` change compilation for the whole process, which helps when debugging. `compile_detailed` and `MlxCompileCache` give direct control over MLX's compile cache.

## Custom differentiation rules

`with_custom_vjp` replaces a closure's gradient, and `with_custom_rules` replaces its VJP, JVP, and VMAP behavior together. Backward rules receive the primals, outputs, and cotangents.

```rust
use mlx_bridge::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxClosure, MlxClosureCustom, install_non_terminating_error_handler, value_and_grad_of_closure};

fn sum_of_squares(inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    Ok(vec![context.sum(&context.square(&inputs[0])?, false)?])
}

fn constant_gradient(
    _primals: &[MlxArray],
    _outputs: &[MlxArray],
    _cotangents: &[MlxArray],
) -> Result<Vec<MlxArray>, MlxCError> {
    Ok(vec![MlxArray::from_scalar_f32(42.0)])
}

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let backward = MlxClosureCustom::from_fn(constant_gradient)?;
    let customized = MlxClosure::from_fn(sum_of_squares)?.with_custom_vjp(&backward)?;
    let value_and_gradient = value_and_grad_of_closure(&customized, &[0])?;

    let point = MlxArray::from_f32(&[2.0], &[1])?;
    let (values, gradients) = value_and_gradient.apply(&MlxArrayVector::from_array(&point)?)?;

    assert_eq!(values.array_at(0, "read")?.item_f32()?, 4.0);
    assert_eq!(gradients.array_at(0, "read")?.item_f32()?, 42.0);
    Ok(())
}
```

## Keyword arguments

`MlxClosureKwargs` receives a name-to-array map alongside its inputs.

```rust
use mlx_bridge::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxClosureKwargs, MlxMapStringToArray, install_non_terminating_error_handler};

fn scaled(inputs: &[MlxArray], keywords: &MlxMapStringToArray) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    let scale = keywords.value_at("scale")?.ok_or(MlxCError {
        operation: "read the scale keyword",
        description: "the scale keyword is required".to_owned(),
    })?;
    Ok(vec![context.multiply(&inputs[0], &scale)?])
}

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let closure = MlxClosureKwargs::from_fn(scaled)?;
    let input = MlxArray::from_f32(&[2.0], &[1])?;
    let mut keywords = MlxMapStringToArray::new()?;
    keywords.insert("scale", &MlxArray::from_scalar_f32(3.0))?;

    let outputs = closure.apply(&MlxArrayVector::from_array(&input)?, &keywords)?;

    assert_eq!(outputs.array_at(0, "read")?.to_vec_f32()?, vec![6.0]);
    Ok(())
}
```

## Failures and panics

A callback's `Err` returns to the caller unchanged, not as MLX's generic callback failure. A panic is caught at the C boundary and becomes an `MlxCError` carrying the message, and the closure stays usable afterward.

```rust
use mlx_bridge::{MlxArray, MlxArrayVector, MlxCError, MlxClosure, install_non_terminating_error_handler};

fn refusing(_inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    Err(MlxCError {
        operation: "run my callback",
        description: "the callback refused".to_owned(),
    })
}

fn panicking(_inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    panic!("the callback panicked")
}

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let no_inputs = MlxArrayVector::empty("build the inputs")?;

    let failure = MlxClosure::from_fn(refusing)?.apply(&no_inputs).expect_err("it refuses");
    assert_eq!(failure.operation, "run my callback");

    let panic_failure = MlxClosure::from_fn(panicking)?.apply(&no_inputs).expect_err("it panics");
    assert_eq!(panic_failure.description, "the callback panicked");
    Ok(())
}
```

A panic still prints through the panic hook. Inside a transform such as `value_and_grad`, the trace also reports failure, though MLX may report its own description for it.

## Which closure type

| Type | Callable shape | Used for |
|---|---|---|
| `MlxClosure` | inputs to outputs | Most transforms |
| `MlxClosureKwargs` | inputs and keywords to outputs | Keyword functions, export |
| `MlxClosureValueAndGrad` | inputs to `(values, gradients)` | The result of `value_and_grad_of_closure` |
| `MlxClosureCustom` | primals, outputs, cotangents | Custom VJP rules |
| `MlxClosureCustomJvp` | primals, tangents, output indices | Custom JVP rules |
| `MlxClosureCustomVmap` | inputs and axes to `(outputs, axes)` | Custom VMAP rules |

For function pointers that build a graph directly on `raw` handles, see `MlxCompiledMultiOutputGraph` in [Model primitives](model-primitives.md#prebuilt-compiled-graphs).
