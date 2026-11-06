# Exporting functions

Exporting traces a function into a file. Importing replays it without the original code, which suits serving a model from a different process, or from another language that has MLX.

## Export and import

`export_function` traces the closure with example arguments. A shape-specific export accepts only those shapes.

```rust
use mlx_c_rust::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxClosure, MlxImportedFunction, export_function, install_non_terminating_error_handler};

fn doubled(inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    Ok(vec![context.multiply_scalar(&inputs[0], 2.0)?])
}

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let path = std::env::temp_dir().join(format!("mlx_docs_{}_doubled.mlxfn", std::process::id()));
    let path = path.to_str().expect("the temporary path is UTF-8");
    let example = MlxArray::from_f32(&[1.0, 2.0], &[2])?;
    let example_arguments = MlxArrayVector::from_array(&example)?;

    export_function(path, &MlxClosure::from_fn(doubled)?, &example_arguments, false)?;
    let imported = MlxImportedFunction::load(path)?;
    std::fs::remove_file(path).expect("the file was just written");

    let outputs = imported.apply(&example_arguments)?;
    assert_eq!(outputs.array_at(0, "read")?.to_vec_f32()?, vec![2.0, 4.0]);
    Ok(())
}
```

## Accepting new shapes

Pass `true` for `shapeless` so the imported function accepts other shapes. This only works for graphs that don't depend on input shape.

To record several specific shapes instead, use `MlxFunctionExporter`: call `apply` once per signature, then drop the exporter to finish the file.

```rust
use mlx_c_rust::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxClosure, MlxFunctionExporter, MlxImportedFunction, install_non_terminating_error_handler};

fn doubled(inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let context = MlxBindingsContext::new()?;
    Ok(vec![context.multiply_scalar(&inputs[0], 2.0)?])
}

fn main() -> Result<(), MlxCError> {
    install_non_terminating_error_handler();
    let path = std::env::temp_dir().join(format!("mlx_docs_{}_traces.mlxfn", std::process::id()));
    let path = path.to_str().expect("the temporary path is UTF-8");
    let short_input = MlxArrayVector::from_array(&MlxArray::from_f32(&[1.0], &[1])?)?;
    let long_input = MlxArrayVector::from_array(&MlxArray::from_f32(&[1.0, 2.0], &[2])?)?;

    let exporter = MlxFunctionExporter::new(path, &MlxClosure::from_fn(doubled)?, false)?;
    exporter.apply(&short_input)?;
    exporter.apply(&long_input)?;
    drop(exporter);

    let imported = MlxImportedFunction::load(path)?;
    std::fs::remove_file(path).expect("the file was just written");

    assert_eq!(imported.apply(&long_input)?.array_at(0, "read")?.to_vec_f32()?, vec![2.0, 4.0]);
    Ok(())
}
```

Keyword closures use `export_function_with_keywords` and `apply_with_keywords`.

## Inspecting graphs

`export_graph_to_dot` writes a Graphviz graph and `print_graph` writes text, both to a C `FILE*` you open and close. `MlxNodeNamer` gives nodes readable names. Both functions are `unsafe` because the compiler cannot check that the file stays open.
